
import json
from pathlib import Path

from src.context_builder import (
    load_documents,
    build_flat_context,
    build_padic_context,
)
from src.retrieval import load_claims, retrieve_claims


ROOT = Path(__file__).resolve().parent.parent


def test_premise_removal():
    with open(ROOT / "data/questions.jsonl", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    question = next(q for q in questions if q["question_id"] == "Q03")

    claims = load_claims()
    documents = load_documents()

    original = retrieve_claims(question["question"], claims)
    original_ids = [c["claim_id"] for c in original]

    removed_id = "C27"

    if not {"C26", "C27"}.issubset(original_ids):
        print("Нужные доказательства отсутствуют в BM25 top-15.")
        print("Найденные ID:", original_ids)
        return

    modified = [
        claim for claim in original
        if claim["claim_id"] != removed_id
    ]

    modified_ids = [c["claim_id"] for c in modified]

    assert len(modified) == len(original) - 1

    output_dir = ROOT / "tests" / "stress_results"
    output_dir.mkdir(exist_ok=True)

    contexts = {
        "B_original": build_flat_context(original),
        "B_modified": build_flat_context(modified),
        "C_original": build_padic_context(original, documents),
        "C_modified": build_padic_context(modified, documents),
    }

    for name, context in contexts.items():
        (output_dir / f"{name}.txt").write_text(
            context, encoding="utf-8"
        )

    result = {
        "test": "premise_removal",
        "question_id": "Q03",
        "removed_claim_id": removed_id,
        "original_claim_ids": original_ids,
        "modified_claim_ids": modified_ids,
        "expected_status": "insufficient",
        "manually_verified": False,
    }

    (output_dir / "premise_removal.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Вопрос:", question["question_id"])
    print("Исходных утверждений:", len(original))
    print("После удаления:", len(modified))
    print("Удалено:", removed_id)
    print("Предполагаемый статус: insufficient")
    print("Результаты сохранены в tests/stress_results/")


if __name__ == "__main__":
    test_premise_removal()
