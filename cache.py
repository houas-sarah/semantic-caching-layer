"""
Cœur de la couche de cache sémantique.

Trois composants collaborent :
- SentenceTransformer : transforme les questions en vecteurs (embeddings), en local.
- FAISS : index vectoriel pour la recherche de similarité. On utilise le produit
  scalaire (Inner Product) sur des vecteurs NORMALISÉS, ce qui équivaut
  exactement à la similarité cosinus.
- Redis : stocke les couples (question -> réponse) et les compteurs de stats.

Les identifiants de vecteurs FAISS et les clés Redis sont synchronisés grâce à
un compteur Redis (`cache:next_id`) et à un `IndexIDMap` FAISS, ce qui garantit
la cohérence même après un redémarrage.
"""

import threading
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import redis
from sentence_transformers import SentenceTransformer

from config import settings

# --- Clés Redis ---
ENTRY_KEY = "cache:entry:{id}"  # hash {question, answer} pour chaque entrée
COUNTER_KEY = "cache:next_id"   # compteur d'identifiants partagé FAISS/Redis
STAT_HITS = "stats:hits"
STAT_MISSES = "stats:misses"
STAT_HIT_LATENCY = "stats:hit_latency_ms_total"
STAT_MISS_LATENCY = "stats:miss_latency_ms_total"


class SemanticCache:
    """Cache sémantique : recherche FAISS + stockage Redis + statistiques."""

    def __init__(self) -> None:
        # Le modèle est téléchargé automatiquement au premier lancement (~90 Mo).
        self.model = SentenceTransformer(settings.embedding_model)
        self.redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
            # RESP2 : compatible avec tous les serveurs Redis, y compris les
            # builds Windows anciens (< 6) qui ne connaissent pas HELLO/RESP3.
            protocol=2,
        )
        # FAISS n'est pas thread-safe en écriture : on protège add/search.
        self._lock = threading.Lock()
        self.index = self._load_or_create_index()

    # ------------------------------------------------------------------ FAISS

    def _load_or_create_index(self) -> faiss.Index:
        """Recharge l'index depuis le disque, ou en crée un nouveau."""
        path = Path(settings.faiss_index_path)
        if path.exists():
            return faiss.read_index(str(path))
        # IndexFlatIP = produit scalaire ; sur des vecteurs normalisés,
        # le score retourné est la similarité cosinus (1.0 = identique).
        # IndexIDMap permet d'attacher nos propres IDs (ceux de Redis).
        return faiss.IndexIDMap(faiss.IndexFlatIP(settings.embedding_dim))

    def _persist_index(self) -> None:
        path = Path(settings.faiss_index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    # ------------------------------------------------------------- Embeddings

    def embed(self, text: str) -> np.ndarray:
        """Retourne l'embedding float32 normalisé de `text`, de forme (1, dim)."""
        vector = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32)

    # ------------------------------------------------------------ Logique cache

    def lookup(self, embedding: np.ndarray) -> Optional[dict]:
        """
        Cherche la question la plus proche dans FAISS.

        Retourne None si l'index est vide, si la meilleure similarité est sous
        le seuil, ou si l'entrée Redis correspondante a disparu. Sinon retourne
        {answer, matched_question, similarity}.
        """
        if self.index.ntotal == 0:
            return None

        with self._lock:
            scores, ids = self.index.search(embedding, k=1)

        similarity, entry_id = float(scores[0][0]), int(ids[0][0])
        if entry_id == -1 or similarity < settings.similarity_threshold:
            return None

        entry = self.redis.hgetall(ENTRY_KEY.format(id=entry_id))
        if not entry:
            return None

        return {
            "answer": entry["answer"],
            "matched_question": entry["question"],
            "similarity": round(similarity, 4),
        }

    def store(self, question: str, answer: str, embedding: np.ndarray) -> None:
        """
        Ajoute une nouvelle entrée au cache (question dans FAISS, couple
        question/réponse dans Redis). Appelé en tâche de fond après un miss,
        pour ne jamais ralentir la réponse renvoyée à l'utilisateur.
        """
        entry_id = self.redis.incr(COUNTER_KEY)
        self.redis.hset(
            ENTRY_KEY.format(id=entry_id),
            mapping={"question": question, "answer": answer},
        )
        with self._lock:
            self.index.add_with_ids(embedding, np.array([entry_id], dtype=np.int64))
            self._persist_index()

    # ------------------------------------------------------------------ Stats

    def record(self, hit: bool, latency_ms: float) -> None:
        """Incrémente les compteurs hit/miss et cumule la latence observée."""
        pipe = self.redis.pipeline()
        pipe.incr(STAT_HITS if hit else STAT_MISSES)
        pipe.incrbyfloat(STAT_HIT_LATENCY if hit else STAT_MISS_LATENCY, latency_ms)
        pipe.execute()

    def stats(self) -> dict:
        """Statistiques agrégées : hit rate, économies estimées, latences moyennes."""
        hits = int(self.redis.get(STAT_HITS) or 0)
        misses = int(self.redis.get(STAT_MISSES) or 0)
        total = hits + misses
        hit_latency_total = float(self.redis.get(STAT_HIT_LATENCY) or 0.0)
        miss_latency_total = float(self.redis.get(STAT_MISS_LATENCY) or 0.0)

        return {
            "total_requests": total,
            "cache_hits": hits,
            "cache_misses": misses,
            "hit_rate": round(hits / total, 4) if total else 0.0,
            "estimated_savings_usd": round(hits * settings.cost_per_llm_call_usd, 4),
            "avg_latency_ms_cache_hit": round(hit_latency_total / hits, 1) if hits else None,
            "avg_latency_ms_llm_call": round(miss_latency_total / misses, 1) if misses else None,
            "cached_entries": self.index.ntotal,
            "similarity_threshold": settings.similarity_threshold,
        }

    def flush(self) -> None:
        """Vide entièrement le cache : index FAISS, entrées Redis et stats."""
        with self._lock:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(settings.embedding_dim))
            self._persist_index()
        self.redis.flushdb()
