"""
Fixtures partagées des tests.

Les tests utilisent la base Redis n°1 (au lieu de 0) et un index FAISS
temporaire : ils ne touchent JAMAIS au cache réel de l'application.
"""

import sys
from pathlib import Path

# Rendre les modules du projet importables depuis tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from config import settings


@pytest.fixture(scope="session")
def cache(tmp_path_factory):
    """Instance SemanticCache unique pour la session (le modèle est long à charger)."""
    settings.redis_db = 1  # base dédiée aux tests
    settings.faiss_index_path = str(tmp_path_factory.mktemp("faiss") / "test.index")

    from cache import SemanticCache

    instance = SemanticCache()
    instance.flush()
    yield instance
    instance.flush()  # ne rien laisser traîner dans la base de test


@pytest.fixture()
def clean_cache(cache):
    """Le même cache, vidé avant chaque test."""
    cache.flush()
    return cache
