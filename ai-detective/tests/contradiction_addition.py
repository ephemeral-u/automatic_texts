
import json
from pathlib import Path

from src.context_builder import (
    load_documents,
    build_full_context,
    build_flat_context,
    build_padic_context,
)
from src.retrieval import load_claims, retrieve_claims


ROOT = Path(__file__).resolve().parent.parent


def run_test():
    documents = load_documents()
    claims = load_claims()

    with open(ROOT / "data/questions.jsonl", encoding="utf-8") as f:
        question = next(
            json.loads(line)
            for line in f
            if json.loads(line)["question_id"] == "Q01"
        )

    new_document = {
        "document_id": "SD01",
        "title": "Альтернативный журнал авторизации",
        "source_type": "system_log",
        "created_at": "2026-03-14T19:10:00",
        "author": "Резервная система мониторинга",
        "text": (
            "Входа Алексея Морозова в 18:23:04 не было; "
            "вход зарегистрирован в 18:25:04."
        ),
    }

    new_claim = {
        "claim_id": "SC01",
        "text": (
            "Алексей Морозов вошёл в административную панель "
            "в 18:25:04, а не в 18:23:04."
        ),
        "document_id": "SD01",
        "evidence": new_document["text"],
        "status": "disputed",
    }

    modified_documents = documents + [new_document]
    modified_claims = claims + [new_claim]

    original = retrieve_claims(question["question"], claims)
    modified = retrieve_claims(question["question"], modified_claims)

    original_ids = [c["claim_id"] for c in original]
    modified_ids = [c["claim_id"] for c in modified]

    assert "C20" in original_ids
    assert {"C20", "SC01"}.issubset(modified_ids)
    assert len(original) == len(modified) == 15
    assert new_claim["evidence"] in new_document["text"]

    contexts = {
        "A_original": build_full_context(documents, claims),
        "A_modified": build_full_context(
            modified_documents, modified_claims
        ),
        "B_original": build_flat_context(original),
        "B_modified": build_flat_context(modified),
        "C_original": build_padic_context(original, documents),
        "C_modified": build_padic_context(
            modified, modified_documents
        ),
    }

    output_dir = ROOT / "tests" / "stress_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, context in contexts.items():
        (output_dir / f"Q01_{name}.txt").write_text(
            context, encoding="utf-8"
        )

    result = {
        "test": "contradiction_addition",
        "question_id": "Q01",
        "added_document": new_document,
        "added_claim": new_claim,
        "original_claim_ids": original_ids,
        "modified_claim_ids": modified_ids,
        "expected_status": "both",
        "manually_verified": False,
    }

    (output_dir / "Q01_contradiction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Вопрос: Q01")
    print("Добавлено утверждение: SC01")
    print("Исходный статус: entailed")
    print("Предполагаемый новый статус: both")
    print("Режим B:", len(modified), "утверждений")
    print("Режим C:", len(modified), "утверждений")
    print("C20 и SC01 присутствуют:", True)
    print("Результаты сохранены")


if __name__ == "__main__":
    run_test()
