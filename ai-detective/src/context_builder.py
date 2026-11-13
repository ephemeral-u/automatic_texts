import re
from src.padic import get_padic_path
import json
from pathlib import Path

from src.retrieval import load_claims, retrieve_claims


ROOT = Path(__file__).resolve().parent.parent


def load_documents():
    path = ROOT / "data" / "documents.jsonl"

    with open(path, encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def build_full_context(documents, claims):
    """Режим A: полное досье."""

    parts = ["ПОЛНОЕ ДОСЬЕ"]

    for doc in documents:
        parts.append(
            f"\nДокумент {doc['document_id']}\n"
            f"Название: {doc['title']}\n"
            f"Дата: {doc['created_at']}\n"
            f"Автор: {doc['author']}\n"
            f"Текст: {doc['text']}\n"
        )

    parts.append("\nИДЕНТИФИКАТОРЫ ДОКАЗАТЕЛЬСТВ")

    for claim in claims:
        parts.append(
            f"{claim['claim_id']} "
            f"({claim['document_id']}): {claim['text']}"
        )

    return "\n".join(parts)


def build_flat_context(claims):
    """Режим B: плоский список утверждений."""

    parts = ["НАЙДЕННЫЕ УТВЕРЖДЕНИЯ"]

    for claim in claims:
        parts.append(
            f"\n{claim['claim_id']}\n"
            f"Документ: {claim['document_id']}\n"
            f"Статус: {claim['status']}\n"
            f"Факт: {claim['text']}\n"
            f"Доказательство: {claim['evidence']}"
        )

    return "\n".join(parts)



def get_event_time(claim):
    """Извлекает время события для сортировки."""
    for text in (claim["evidence"], claim["text"]):
        match = re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", text)
        if match:
            return match.group(0)

    return "99:99:99"


def build_padic_context(claims, documents):
    """Режим C: группировка утверждений по p-adic ветвям."""

    documents_by_id = {
        doc["document_id"]: doc for doc in documents
    }

    branches = {}

    for claim in claims:
        document = documents_by_id[claim["document_id"]]
        path = get_padic_path(claim, document)

        branches.setdefault(path, []).append(claim)

    parts = ["P-ADIC КОНТЕКСТ"]

    for path, branch_claims in sorted(branches.items()):
        parts.append(f"\nВЕТВЬ {list(path)}")

        ordered = sorted(branch_claims, key=get_event_time)

        for claim in ordered:
            parts.append(
                f"\n{claim['claim_id']}\n"
                f"Документ: {claim['document_id']}\n"
                f"Статус: {claim['status']}\n"
                f"Факт: {claim['text']}\n"
                f"Доказательство: {claim['evidence']}"
            )

    return "\n".join(parts)


if __name__ == "__main__":
    documents = load_documents()
    claims = load_claims()

    question = (
        "Во сколько Алексей Морозов вошёл "
        "в административную панель?"
    )

    selected_claims = retrieve_claims(question, claims)

    context_a = build_full_context(documents, claims)
    context_b = build_flat_context(selected_claims)

    print("РЕЖИМ A")
    print("Документов:", len(documents))
    print("Символов:", len(context_a))

    print("\nРЕЖИМ B")
    print("Утверждений:", len(selected_claims))
    print("Символов:", len(context_b))

    print("\nВыбранные claim_id:")
    print([c["claim_id"] for c in selected_claims])
    context_c = build_padic_context(selected_claims, documents)

    print("\nРЕЖИМ C")
    print("Утверждений:", len(selected_claims))
    print("Символов:", len(context_c))

    ids_b = set(re.findall(r"\bC\d{2}\b", context_b))
    ids_c = set(re.findall(r"\bC\d{2}\b", context_c))

    print("Одинаковые claim_id:", ids_b == ids_c)

    print("\nP-ADIC ВЕТВИ:")
    print(context_c)
