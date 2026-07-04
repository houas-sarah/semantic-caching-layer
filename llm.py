"""
Client LLM, appelé uniquement en cas de Cache Miss.

Deux fournisseurs gratuits sont supportés, sélectionnés via LLM_PROVIDER :
- "groq"   : API Groq (https://console.groq.com/keys) — très rapide.
- "gemini" : API Google Gemini (https://aistudio.google.com/apikey).

Les clients sont créés paresseusement (au premier appel) et mis en cache,
pour ne pas exiger la clé API de l'autre fournisseur.
"""

from functools import lru_cache

from config import settings


@lru_cache(maxsize=1)
def _groq_client():
    from groq import Groq

    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY manquante : ajoutez votre clé (gratuite) dans le "
            "fichier .env — https://console.groq.com/keys"
        )
    return Groq(api_key=settings.groq_api_key)


@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai

    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY manquante : ajoutez votre clé (gratuite) dans le "
            "fichier .env — https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=settings.gemini_api_key)


def ask_llm(question: str) -> str:
    """Envoie `question` au fournisseur configuré et retourne le texte de la réponse."""
    if settings.llm_provider == "groq":
        response = _groq_client().chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": question}],
        )
        return response.choices[0].message.content

    if settings.llm_provider == "gemini":
        response = _gemini_client().models.generate_content(
            model=settings.gemini_model,
            contents=question,
        )
        return response.text

    raise ValueError(
        f"LLM_PROVIDER inconnu : {settings.llm_provider!r} (attendu 'groq' ou 'gemini')"
    )
