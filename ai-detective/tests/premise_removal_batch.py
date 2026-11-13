
import json
from pathlib import Path

from src.context_builder import (
    load_documents,
    build_flat_context,
    build_padic_context,
)
from src.retrieval import load_claims, retrieve_claims


ROOT = Path(__file__).resolve().parent.parent

REMOVALS = {
    "Q04": "C19",
    "Q05": "C20",
    "Q06": "C24",
}


def run_tests():
    questions_path = ROOT / "data" / "questions.jsonl"

    with open(questions_path, encoding="utf-8") as f:
        questions = {
            q["question_id"]: q
            for line in f
            if (q := json.loads(line))
        }

    claims = load_claims()
    documents = load_documents()

    output_dir = ROOT / "tests" / "stress_results"
    output_dir.mkdir(exist_ok=True)

    for question_id, removed_id in REMOVALS.items():
        question = questions[question_id]

        original = retrieve_claims(question["question"], claims)
        original_ids = [c["claim_id"] for c in original]

        required_ids = set(question["gold_evidence_ids"])
        missing_ids = required_ids - set(original_ids)

        print(f"\nВопрос: {question_id}")

        if missing_ids:
            print("Пропущен: BM25 не нашёл все эталонные посылки.")
            print("Отсутствуют:", sorted(missing_ids))
            continue

        if removed_id not in original_ids:
            print("Пропущен: удаляемое утверждение отсутствует.")
            continue

        modified = [
            c for c in original
            if c["claim_id"] != removed_id
        ]

        contexts = {
            "B_original": build_flat_context(original),
            "B_modified": build_flat_context(modified),
            "C_original": build_padic_context(original, documents),
            "C_modified": build_padic_context(modified, documents),
        }

        for name, context in contexts.items():
            path = output_dir / f"{question_id}_{name}.txt"
            path.write_text(context, encoding="utf-8")

        result = {
            "question_id": question_id,
            "removed_claim_id": removed_id,
            "original_claim_ids": original_ids,
            "modified_claim_ids": [
                c["claim_id"] for c in modified
            ],
            "expected_status": None,
            "manually_verified": False,
        }

        path = output_dir / f"{question_id}_removal.json"
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print("Исходных утверждений:", len(original))
        print("После удаления:", len(modified))
        print("Удалено:", removed_id)
        print("Контексты сохранены.")
        print("Ожидаемый статус требует проверки.")


if __name__ == "__main__":
    run_tests()
