"""
Configuration centralisée du projet.

Toutes les valeurs peuvent être surchargées via des variables d'environnement
ou un fichier `.env` (voir `.env.example`).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Embeddings (calculés en local, gratuits) ---
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384  # dimension des vecteurs produits par all-MiniLM-L6-v2

    # Seuil de similarité cosinus [0..1] au-dessus duquel on considère
    # deux questions comme équivalentes. Plus haut = plus strict.
    similarity_threshold: float = 0.85

    # --- Redis (stockage question -> réponse + statistiques) ---
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # --- FAISS (index vectoriel persisté sur disque) ---
    faiss_index_path: str = "data/faiss.index"

    # --- LLM appelé uniquement en cas de Cache Miss ---
    llm_provider: str = "groq"  # "groq" ou "gemini"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # --- Statistiques ---
    # Coût simulé d'un appel LLM, utilisé pour estimer les économies.
    cost_per_llm_call_usd: float = 0.01


settings = Settings()
