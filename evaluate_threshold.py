"""
Évaluation quantitative du seuil de similarité sur un vrai dataset :
Quora Question Pairs (404 000 paires de questions étiquetées duplicate/non-duplicate).

Pour chaque seuil candidat, on mesure :
- précision : parmi les paires prédites "duplicate" (sim >= seuil), combien le sont vraiment.
  -> Une précision faible = le cache servirait de MAUVAISES réponses.
- rappel : parmi les vrais duplicates, combien sont détectés.
  -> Un rappel faible = des hits manqués (simple surcoût, pas d'erreur).

Pour un cache sémantique, LA PRÉCISION PRIME : un faux positif sert une réponse
hors sujet à l'utilisateur, alors qu'un faux négatif coûte juste un appel LLM.

Usage : python evaluate_threshold.py
(Télécharge ~55 Mo au premier lancement. Si l'URL ne répond plus, récupérez le
fichier "quora_duplicate_questions.tsv" sur Kaggle : dataset "Quora Question Pairs".)
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from config import settings

DATASET_URL = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
DATASET_PATH = Path("data/quora_duplicate_questions.tsv")
SAMPLE_PER_CLASS = 2000  # paires par classe (duplicate / non-duplicate)
THRESHOLDS = [round(0.70 + 0.025 * i, 3) for i in range(11)]  # 0.70 -> 0.95
SEED = 42


def download_dataset() -> None:
    if DATASET_PATH.exists():
        return
    print(f"Téléchargement du dataset (~55 Mo) vers {DATASET_PATH}...")
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(DATASET_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(DATASET_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    except Exception as exc:
        sys.exit(
            f"Échec du téléchargement ({exc}).\n"
            f"Récupérez 'quora_duplicate_questions.tsv' sur Kaggle (Quora Question "
            f"Pairs) et placez-le dans {DATASET_PATH}."
        )


def load_pairs() -> tuple[list[tuple[str, str]], list[int]]:
    """Échantillonne SAMPLE_PER_CLASS paires de chaque classe, de façon reproductible."""
    duplicates, non_duplicates = [], []
    with open(DATASET_PATH, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            try:
                pair = (row["question1"], row["question2"])
                label = int(row["is_duplicate"])
            except (KeyError, TypeError, ValueError):
                continue  # lignes malformées du TSV
            (duplicates if label == 1 else non_duplicates).append(pair)

    rng = random.Random(SEED)
    duplicates = rng.sample(duplicates, min(SAMPLE_PER_CLASS, len(duplicates)))
    non_duplicates = rng.sample(non_duplicates, min(SAMPLE_PER_CLASS, len(non_duplicates)))

    pairs = duplicates + non_duplicates
    labels = [1] * len(duplicates) + [0] * len(non_duplicates)
    return pairs, labels


def main() -> None:
    download_dataset()
    pairs, labels = load_pairs()
    print(f"{len(pairs)} paires échantillonnées ({sum(labels)} duplicates).")

    print(f"Calcul des embeddings ({settings.embedding_model})...")
    model = SentenceTransformer(settings.embedding_model)
    q1 = model.encode([p[0] for p in pairs], normalize_embeddings=True,
                      batch_size=64, show_progress_bar=True)
    q2 = model.encode([p[1] for p in pairs], normalize_embeddings=True,
                      batch_size=64, show_progress_bar=True)
    similarities = np.sum(q1 * q2, axis=1)  # cosinus, paire par paire
    labels_arr = np.array(labels)

    print(f"\n{'SEUIL':>6} {'PRÉCISION':>10} {'RAPPEL':>8} {'F1':>7}   (hit = sim >= seuil)")
    print("-" * 50)
    best = None
    for threshold in THRESHOLDS:
        predicted = similarities >= threshold
        tp = int(np.sum(predicted & (labels_arr == 1)))
        fp = int(np.sum(predicted & (labels_arr == 0)))
        fn = int(np.sum(~predicted & (labels_arr == 1)))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        marker = ""
        if best is None or f1 > best[1]:
            best = (threshold, f1)
        print(f"{threshold:>6} {precision:>10.3f} {recall:>8.3f} {f1:>7.3f}")

    print(f"\nMeilleur F1 : seuil {best[0]} (F1={best[1]:.3f})")
    print(
        "Rappel : pour un cache, privilégiez un seuil où la PRÉCISION est élevée\n"
        "(>= 0.95) même au prix d'un rappel plus faible — un faux positif sert une\n"
        "mauvaise réponse, un faux négatif ne coûte qu'un appel LLM."
    )


if __name__ == "__main__":
    main()
