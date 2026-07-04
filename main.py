"""
API FastAPI : point d'entrée de la couche de cache sémantique.

Endpoints :
- POST   /ask    : répond à une question (cache sémantique, sinon LLM).
- GET    /stats  : hit rate, économies estimées, latences moyennes.
- GET    /health : état de l'API et de Redis.
- DELETE /cache  : vide entièrement le cache.

Lancement : uvicorn main:app --reload
"""

import time
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from cache import SemanticCache
from llm import ask_llm

cache: SemanticCache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle d'embeddings et l'index FAISS une seule fois, au démarrage."""
    global cache
    cache = SemanticCache()
    yield


app = FastAPI(
    title="Semantic Caching Layer",
    description="Cache sémantique pour LLM : réduit la latence et les coûts d'API "
    "en servant depuis le cache les questions déjà vues (même reformulées).",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        examples=["Quelle est la capitale de la France ?"],
    )


@app.post("/ask")
def ask(request: AskRequest, background_tasks: BackgroundTasks):
    """
    Répond à une question en suivant la logique du cache sémantique :

    1. Embedding local de la question (sentence-transformers).
    2. Recherche du plus proche voisin dans FAISS.
    3. HIT  (similarité >= seuil) : réponse servie depuis Redis, sans appel LLM.
    4. MISS : appel du LLM, réponse renvoyée immédiatement, puis stockage
       (FAISS + Redis) en tâche de fond.
    """
    start = time.perf_counter()

    embedding = cache.embed(request.question)
    match = cache.lookup(embedding)

    if match:  # ---------------------------------------------------- Cache HIT
        latency_ms = (time.perf_counter() - start) * 1000
        cache.record(hit=True, latency_ms=latency_ms)
        return {
            "source": "cache",
            "answer": match["answer"],
            "matched_question": match["matched_question"],
            "similarity": match["similarity"],
            "latency_ms": round(latency_ms, 1),
        }

    # ------------------------------------------------------------- Cache MISS
    try:
        answer = ask_llm(request.question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Échec de l'appel LLM : {exc}")
    latency_ms = (time.perf_counter() - start) * 1000
    cache.record(hit=False, latency_ms=latency_ms)

    # Le stockage se fait APRÈS l'envoi de la réponse : l'utilisateur
    # ne paie jamais le coût de l'écriture dans le cache.
    background_tasks.add_task(cache.store, request.question, answer, embedding)

    return {
        "source": "llm",
        "answer": answer,
        "latency_ms": round(latency_ms, 1),
    }


@app.get("/stats")
def stats():
    """Taux de réussite du cache et estimation des économies réalisées."""
    return cache.stats()


@app.get("/health")
def health():
    """Vérifie que l'API et Redis répondent."""
    return {
        "status": "ok",
        "redis_connected": cache.redis.ping(),
        "cached_entries": cache.index.ntotal,
    }


@app.delete("/cache")
def clear_cache():
    """Vide l'index FAISS, les entrées Redis et les statistiques."""
    cache.flush()
    return {"status": "cache cleared"}
