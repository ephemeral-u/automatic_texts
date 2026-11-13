
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parent.parent


def load_claims():
    path = ROOT / "data" / "claims.jsonl"

    with open(path, encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def tokenize(text):
    return re.findall(r"[^\W_]+", text.lower())


def retrieve_claims(question, claims, top_k=15):
    if not claims or top_k <= 0:
        return []

    corpus = [
        tokenize(claim["text"])
        for claim in claims
    ]

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(question))

    ranked = sorted(
        range(len(claims)),
        key=lambda i: (-scores[i], i)
    )

    return [
        claims[i]
        for i in ranked[:min(top_k, 15)]
    ]


if __name__ == "__main__":
    claims = load_claims()

    question = (
        "Во сколько Алексей Морозов вошёл "
        "в административную панель?"
    )

    results = retrieve_claims(question, claims)

    print(f"Вопрос: {question}")
    print(f"Найдено утверждений: {len(results)}\n")

    for claim in results:
        print(
            f"{claim['claim_id']}: {claim['text']}"
        )
