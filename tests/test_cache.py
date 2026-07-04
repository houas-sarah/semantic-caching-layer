"""Tests du cœur du cache sémantique : embeddings, FAISS, Redis, stats, persistance."""

import numpy as np

from config import settings

QUESTION = "Quelle est la capitale de la France ?"
ANSWER = "Paris est la capitale de la France."
PARAPHRASE = "C'est quoi la capitale de la France ?"
UNRELATED = "Comment faire une pizza margherita ?"


def test_embedding_shape_and_normalization(cache):
    emb = cache.embed("Bonjour le monde")
    assert emb.shape == (1, settings.embedding_dim)
    assert emb.dtype == np.float32
    # Vecteur normalisé => produit scalaire == similarité cosinus
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_empty_index_returns_none(clean_cache):
    assert clean_cache.lookup(clean_cache.embed(QUESTION)) is None


def test_exact_match_is_a_hit(clean_cache):
    emb = clean_cache.embed(QUESTION)
    clean_cache.store(QUESTION, ANSWER, emb)

    match = clean_cache.lookup(clean_cache.embed(QUESTION))
    assert match is not None
    assert match["answer"] == ANSWER
    assert match["similarity"] >= 0.999


def test_paraphrase_above_threshold_is_a_hit(clean_cache):
    clean_cache.store(QUESTION, ANSWER, clean_cache.embed(QUESTION))

    match = clean_cache.lookup(clean_cache.embed(PARAPHRASE))
    assert match is not None
    assert match["matched_question"] == QUESTION
    assert match["similarity"] >= settings.similarity_threshold


def test_unrelated_question_is_a_miss(clean_cache):
    clean_cache.store(QUESTION, ANSWER, clean_cache.embed(QUESTION))
    assert clean_cache.lookup(clean_cache.embed(UNRELATED)) is None


def test_missing_redis_entry_is_a_miss(clean_cache):
    """Si l'entrée Redis a disparu (éviction), le match FAISS doit être ignoré."""
    clean_cache.store(QUESTION, ANSWER, clean_cache.embed(QUESTION))
    clean_cache.redis.delete("cache:entry:1")

    assert clean_cache.lookup(clean_cache.embed(QUESTION)) is None


def test_stats_tracking(clean_cache):
    clean_cache.record(hit=True, latency_ms=10.0)
    clean_cache.record(hit=False, latency_ms=100.0)

    stats = clean_cache.stats()
    assert stats["total_requests"] == 2
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["estimated_savings_usd"] == settings.cost_per_llm_call_usd
    assert stats["avg_latency_ms_cache_hit"] == 10.0
    assert stats["avg_latency_ms_llm_call"] == 100.0


def test_persistence_across_restarts(clean_cache):
    """L'index FAISS est persisté : une nouvelle instance doit retrouver les entrées."""
    from cache import SemanticCache

    clean_cache.store(QUESTION, ANSWER, clean_cache.embed(QUESTION))

    reloaded = SemanticCache()  # relit l'index depuis le disque + même Redis
    match = reloaded.lookup(reloaded.embed(QUESTION))
    assert match is not None
    assert match["answer"] == ANSWER
