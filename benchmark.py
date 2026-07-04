"""
Benchmark de démonstration : envoie une question puis plusieurs reformulations,
et affiche la différence de latence entre appel LLM (miss) et cache (hit).

Prérequis : l'API doit tourner (uvicorn main:app) et Redis être démarré.
Usage : python benchmark.py
"""

import os
import time

import requests

# Surchargez avec API_URL=http://127.0.0.1:8001 si le port 8000 est occupé.
# NB : on utilise 127.0.0.1 et non localhost — sous Windows, localhost tente
# d'abord IPv6 et ajoute ~2 s de latence artificielle à chaque requête.
API = os.environ.get("API_URL", "http://127.0.0.1:8000")

QUESTIONS = [
    "Quelle est la capitale de la France ?",          # MISS attendu (1er appel)
    "Quelle est la capitale de la France ?",          # HIT (question identique)
    "C'est quoi la capitale de la France ?",          # HIT (paraphrase)
    "Peux-tu me dire quelle ville est la capitale française ?",  # HIT (sémantique)
    "Quelle est la capitale de l'Allemagne ?",        # MISS (question différente)
]


def main() -> None:
    print(f"{'SOURCE':<8} {'LATENCE':>10}   QUESTION")
    print("-" * 80)

    for question in QUESTIONS:
        start = time.perf_counter()
        response = requests.post(f"{API}/ask", json={"question": question}).json()
        elapsed_ms = (time.perf_counter() - start) * 1000

        similarity = response.get("similarity")
        extra = f" (similarité: {similarity})" if similarity is not None else ""
        print(f"{response['source']:<8} {elapsed_ms:>8.1f} ms   {question}{extra}")

    print("\nStatistiques globales (/stats) :")
    for key, value in requests.get(f"{API}/stats").json().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
