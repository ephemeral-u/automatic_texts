
import json
import random
from pathlib import Path

from src.context_builder import load_documents, build_full_context
from src.retrieval import load_claims


ROOT = Path(__file__).resolve().parent.parent


def load_questions():
    with open(ROOT / "data" / "questions.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_document_shuffle():
    documents = load_documents()
    claims = load_claims()
    questions = load_questions()

    original_ids = [d["document_id"] for d in documents]

    shuffled = documents.copy()
    random.Random(42).shuffle(shuffled)

    shuffled_ids = [d["document_id"] for d in shuffled]

    assert sorted(original_ids) == sorted(shuffled_ids)
    assert original_ids != shuffled_ids

    original_context = build_full_context(documents, claims)
    shuffled_context = build_full_context(shuffled, claims)

    assert original_context != shuffled_context

    result = {
        "test": "document_shuffle",
        "random_state": 42,
        "original_order": original_ids,
        "modified_order": shuffled_ids,
        "expected_statuses": {
            q["question_id"]: q["gold_status"]
            for q in questions
        }
    }

    output = ROOT / "tests" / "permutation_manifest.json"

    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Исходный порядок:", original_ids)
    print("Новый порядок:", shuffled_ids)
    print("Состав документов сохранён:", True)
    print("Вопросов:", len(questions))
    print("Результат сохранён:", output.name)


if __name__ == "__main__":
    test_document_shuffle()
