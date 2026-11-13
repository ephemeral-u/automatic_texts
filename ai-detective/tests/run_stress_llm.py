
import json
import re
from pathlib import Path

from src.llm import ask_llm


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "tests" / "stress_results"

QUESTION = "Во сколько Алексей Морозов вошёл в административную панель?"


def run_test():
    for mode in ["B_original", "B_modified"]:
        path = RESULTS / f"Q01_{mode}.txt"
        context = path.read_text(encoding="utf-8")

        claim_ids = set(
            re.findall(r"(?m)^(?:C\d{2}|SC\d{2})$", context)
        )

        print(f"\nРежим: {mode}")

        try:
            answer, metrics = ask_llm(
                question_id="Q01",
                question=QUESTION,
                context=context,
                available_claim_ids=claim_ids
            )

            result = {
                "mode": mode,
                "answer": answer.model_dump(mode="json"),
                "metrics": metrics
            }

            print("Статус:", answer.status.value)
            print("Ответ:", answer.answer)
            print("Confidence:", answer.confidence)
            print("Метрики:", metrics)

        except Exception as error:
            result = {
                "mode": mode,
                "error": str(error)
            }
            print("Ошибка:", error)

        output = RESULTS / f"Q01_{mode}_result.json"
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )


if __name__ == "__main__":
    run_test()
