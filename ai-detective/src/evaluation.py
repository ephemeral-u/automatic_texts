
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path):
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def calculate_f1(predicted, expected):
    predicted = set(predicted)
    expected = set(expected)

    if not predicted and not expected:
        return 1.0

    tp = len(predicted & expected)

    precision = tp / len(predicted) if predicted else 0
    recall = tp / len(expected) if expected else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return f1


def main():
    print("\n========== МЕТРИКИ И АНАЛИЗ ==========\n")

    questions = {
        q["question_id"]: q
        for q in load_jsonl(ROOT / "data/questions.jsonl")
    }

    results = load_jsonl(ROOT / "tests/llm_results.jsonl")

    stress_dir = ROOT / "tests/stress_results"

    for name in ["Q01_B_original", "Q01_B_modified"]:
        path = stress_dir / f"{name}_result.json"

        if path.exists():
            with open(path, encoding="utf-8") as f:
                result = json.load(f)

            if "answer" in result:
                result["question_id"] = "Q01"
                results.append(result)

    print("Найдено ответов LLM:", len(results))

    if not results:
        print("Сохранённых результатов пока нет.")
        return

    correct = 0
    evaluated = 0
    f1_scores = []
    counter_recalls = []
    confident_errors = 0
    total_tokens = 0
    total_time = 0
    status_results = []

    for result in results:
        qid = result["question_id"]
        mode = result.get("mode", "B")
        answer = result["answer"]
        metrics = result["metrics"]

        if qid not in questions:
            continue

        gold = questions[qid]
        gold_status = gold["gold_status"]
        gold_ids = gold["gold_evidence_ids"]
        gold_counter = gold["gold_counterevidence_ids"]

        scenario = result.get("stress_key", "")

        if mode == "B_modified" or (
            qid == "Q01" and "противоречия" in scenario
        ):
            gold_status = "both"
            gold_ids = ["C20"]
            gold_counter = ["SC01"]

        elif "Удаление посылки" in scenario:
            gold_status = None

        elif "Перестановка" in scenario:
            pass

        status = answer["status"]
        confidence = answer["confidence"]

        f1 = calculate_f1(answer["evidence_ids"], gold_ids)
        f1_scores.append(f1)

        if gold_counter:
            found = len(
                set(answer["counterevidence_ids"]) & set(gold_counter)
            )
            counter_recalls.append(found / len(set(gold_counter)))

        if gold_status is not None:
            evaluated += 1
            is_correct = status == gold_status
            correct += int(is_correct)

            if not is_correct and confidence >= 0.8:
                confident_errors += 1

            status_results.append((gold_status, status))

        total_tokens += (
            metrics["input_tokens"] + metrics["output_tokens"]
        )
        total_time += metrics["duration_seconds"]

        print(f"\nВопрос: {qid} | Режим: {mode}")
        print("Статус модели:", status)
        print("Эталон:", gold_status or "Не проверен")
        print("Confidence:", confidence)
        print("Evidence F1:", round(f1, 3))
        print("Время:", metrics["duration_seconds"], "с")
        print(
            "Токены:",
            metrics["input_tokens"] + metrics["output_tokens"]
        )

    print("\n========== ОБЩИЕ МЕТРИКИ ==========\n")

    if evaluated:
        print("Accuracy:", round(correct / evaluated, 3))
        print("Проверенных ответов:", evaluated)
        print("Правильных статусов:", correct)
        print("Уверенных ошибок:", confident_errors)
    else:
        print("Accuracy: нет проверенных эталонов")

    print(
        "Средний Evidence F1:",
        round(sum(f1_scores) / len(f1_scores), 3)
    )

    if counter_recalls:
        print(
            "Counterevidence Recall:",
            round(sum(counter_recalls) / len(counter_recalls), 3)
        )
    else:
        print("Counterevidence Recall: нет данных")

    for target in ["both", "insufficient"]:
        matching = [
            predicted
            for gold, predicted in status_results
            if gold == target
        ]

        if matching:
            recall = matching.count(target) / len(matching)
            print(f"Распознавание {target}:", round(recall, 3))
        else:
            print(f"Распознавание {target}: нет данных")

    print("Всего токенов:", total_tokens)
    print("Общее время:", round(total_time, 2), "с")
    print(
        "Среднее время:",
        round(total_time / len(results), 2), "с"
    )

    print("\n========== ВЫВОД ==========\n")

    if evaluated:
        print(
            f"Модель правильно определила "
            f"{correct} из {evaluated} проверенных статусов."
        )

    if confident_errors:
        print(
            f"Обнаружено {confident_errors} ошибок "
            f"с confidence >= 0.8."
        )

    print("Анализ завершён.")


if __name__ == "__main__":
    main()
