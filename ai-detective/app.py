from datetime import datetime
from src.llm import ask_llm
import json
from pathlib import Path
import random
import streamlit as st
import pandas as pd

from src.context_builder import (
    load_documents,
    build_full_context,
    build_flat_context,
    build_padic_context,
)
from src.retrieval import load_claims, retrieve_claims


ROOT = Path(__file__).resolve().parent


def load_questions():
    with open(ROOT / "data" / "questions.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


st.set_page_config(page_title="AI-детектив", layout="wide")
st.title("AI-детектив")
st.caption("Исследование ответов языковой модели Qwen")


def read_jsonl(uploaded_file, default_loader):
    if uploaded_file is None:
        return default_loader()

    text = uploaded_file.getvalue().decode("utf-8")

    return [
        json.loads(line)
        for line in text.splitlines()
        if line.strip()
    ]


st.sidebar.subheader("Загрузка досье")

uploaded_documents = st.sidebar.file_uploader(
    "documents.jsonl", type="jsonl"
)

uploaded_claims = st.sidebar.file_uploader(
    "claims.jsonl", type="jsonl"
)

uploaded_questions = st.sidebar.file_uploader(
    "questions.jsonl", type="jsonl"
)

try:
    documents = read_jsonl(uploaded_documents, load_documents)
    claims = read_jsonl(uploaded_claims, load_claims)
    questions = read_jsonl(uploaded_questions, load_questions)

except (ValueError, UnicodeDecodeError, KeyError) as error:
    st.error(f"Ошибка загрузки данных: {error}")
    st.stop()


st.sidebar.header("Досье")
st.sidebar.write(f"Документов: {len(documents)}")
st.sidebar.write(f"Утверждений: {len(claims)}")
st.sidebar.write(f"Вопросов: {len(questions)}")

question_ids = [q["question_id"] for q in questions]

selected_id = st.selectbox(
    "Выберите вопрос",
    question_ids
)

question = next(
    q for q in questions
    if q["question_id"] == selected_id
)

st.write("**Вопрос:**", question["question"])

mode = st.radio(
    "Режим анализа",
    ["A — полный контекст", "B — BM25", "C — p-adic"],
    horizontal=True
)


selected_claims = retrieve_claims(question["question"], claims)

test_documents = documents.copy()
test_claims = claims.copy()

stress_options = ["Без изменений"]

if mode.startswith("A"):
    stress_options.append("Перестановка документов")
elif selected_id in ("Q03", "Q04"):
    stress_options.append("Удаление посылки")

if selected_id == "Q01":
    stress_options.append("Добавление противоречия")

stress = st.selectbox("Стресс-тест", stress_options)
removed_id = None

if stress == "Перестановка документов":
    random.Random(42).shuffle(test_documents)

elif stress == "Удаление посылки":
    available = [
        c["claim_id"] for c in selected_claims
        if c["claim_id"] in question["gold_evidence_ids"]
    ]

    removed_id = st.selectbox(
        "Удаляемое доказательство", available
    )

    selected_claims = [
        c for c in selected_claims
        if c["claim_id"] != removed_id
    ]

elif stress == "Добавление противоречия":
    new_doc = {
        "document_id": "SD01",
        "title": "Альтернативный журнал",
        "source_type": "system_log",
        "created_at": "2026-03-14T19:10:00",
        "author": "Резервная система мониторинга",
        "text": (
            "Входа Алексея Морозова в 18:23:04 не было; "
            "вход зарегистрирован в 18:25:04."
        )
    }

    new_claim = {
        "claim_id": "SC01",
        "text": (
            "Алексей Морозов вошёл в административную панель "
            "в 18:25:04, а не в 18:23:04."
        ),
        "document_id": "SD01",
        "evidence": new_doc["text"],
        "status": "disputed"
    }

    test_documents.append(new_doc)
    test_claims.append(new_claim)

    selected_claims = retrieve_claims(
        question["question"], test_claims
    )

if mode.startswith("A"):
    context = build_full_context(test_documents, test_claims)
    used_claims = test_claims

elif mode.startswith("B"):
    context = build_flat_context(selected_claims)
    used_claims = selected_claims

else:
    context = build_padic_context(
        selected_claims, test_documents
    )
    used_claims = selected_claims

stress_key = (
    f"{stress}:{removed_id}" if removed_id else stress
)


st.subheader("Подготовленный контекст")

st.text_area(
    "Контекст для LLM",
    value=context,
    height=350
)

with st.expander("Эталонный ответ"):
    st.write("Статус:", question["gold_status"])
    st.write("Ответ:", question["gold_answer"])
    st.write("Доказательства:", question["gold_evidence_ids"])
    st.write("Контрдоказательства:", question["gold_counterevidence_ids"])


st.divider()
st.subheader("Ответ языковой модели")

run_key = (selected_id, mode[0], stress_key)

if st.button("Получить ответ", type="primary"):
    with st.spinner("Qwen анализирует документы..."):
        try:
            answer, metrics = ask_llm(
                question_id=selected_id,
                question=question["question"],
                context=context,
                available_claim_ids={
                    c["claim_id"] for c in used_claims
                }
            )

            result = {
                "created_at": datetime.now().isoformat(),
                "question_id": selected_id,
                "mode": mode[0],
                "answer": answer.model_dump(mode="json"),
                "stress_key": stress_key,
                "metrics": metrics
            }

            results_path = ROOT / "tests" / "llm_results.jsonl"
            results_path.parent.mkdir(exist_ok=True)

            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            st.session_state["last_result"] = result

        except Exception as error:
            st.error(f"Ошибка: {error}")

result = st.session_state.get("last_result")

if result and (
    result["question_id"],
    result["mode"],
    result.get("stress_key", "Без изменений")
) == run_key:

    answer = result["answer"]
    metrics = result["metrics"]

    st.write("**Статус:**", answer["status"])
    st.write("**Ответ:**", answer["answer"])
    st.write("**Confidence:**", answer["confidence"])
    st.write("**Объяснение:**", answer["explanation"])

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Доказательства:**")
        for claim_id in answer["evidence_ids"]:
            claim = next(
                (c for c in claims if c["claim_id"] == claim_id),
                None
            )
            if claim:
                st.write(f"{claim_id}: {claim['text']}")

    with col2:
        st.write("**Контрдоказательства:**")
        for claim_id in answer["counterevidence_ids"]:
            claim = next(
                (c for c in claims if c["claim_id"] == claim_id),
                None
            )
            if claim:
                st.write(f"{claim_id}: {claim['text']}")

    st.subheader("Метрики")

    col1, col2, col3 = st.columns(3)

    col1.metric("Входные токены", metrics["input_tokens"])
    col2.metric("Выходные токены", metrics["output_tokens"])
    col3.metric("Время, с", metrics["duration_seconds"])

    correct = answer["status"] == question["gold_status"]

    if correct:
        st.success("Логический статус совпадает с эталоном")
    else:
        st.warning("Логический статус не совпадает с эталоном")



st.divider()
st.subheader("Сравнение сохранённых результатов")


def load_saved_results():
    results = []

    # Результаты из Streamlit
    path = ROOT / "tests" / "llm_results.jsonl"

    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

    # Результаты отдельного стресс-теста
    stress_dir = ROOT / "tests" / "stress_results"

    if stress_dir.exists():
        for path in stress_dir.glob("Q*_B_*_result.json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            if "answer" not in data:
                continue

            scenario = (
                "Добавление противоречия"
                if "modified" in data["mode"]
                else "Без изменений"
            )

            results.append({
                "question_id": path.name.split("_")[0],
                "mode": "B",
                "stress_key": scenario,
                "answer": data["answer"],
                "metrics": data["metrics"],
                "created_at": ""
            })

    return results


saved_results = load_saved_results()

if not saved_results:
    st.info("Сохранённых результатов пока нет.")

else:
    rows = []

    for item in saved_results:
        q = next(
            (q for q in questions
             if q["question_id"] == item["question_id"]),
            None
        )

        if q is None:
            continue

        scenario = item.get("stress_key", "Без изменений")

        gold_status = q["gold_status"]

        if (
            item["question_id"] == "Q01"
            and scenario == "Добавление противоречия"
        ):
            gold_status = "both"

        # Для остальных изменённых сценариев
        # эталон пока не подтверждён
        elif scenario != "Без изменений":
            gold_status = None

        answer = item["answer"]
        metrics = item["metrics"]

        rows.append({
            "Вопрос": item["question_id"],
            "Режим": item["mode"],
            "Стресс-тест": scenario,
            "Статус LLM": answer["status"],
            "Эталон": gold_status or "Не проверен",
            "Совпадение": (
                "Да" if answer["status"] == gold_status
                else "Нет" if gold_status else "—"
            ),
            "Confidence": answer["confidence"],
            "Входные токены": metrics["input_tokens"],
            "Выходные токены": metrics["output_tokens"],
            "Время, с": metrics["duration_seconds"]
        })

    if rows:
        df = pd.DataFrame(rows)

        selected_results = df[
            df["Вопрос"] == selected_id
        ]

        st.write(f"Сохранённые результаты для {selected_id}")

        st.dataframe(
            selected_results,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Общая статистика")

        col1, col2, col3 = st.columns(3)

        col1.metric("Всего запусков", len(df))

        checked = df[df["Совпадение"] != "—"]

        accuracy = (
            (checked["Совпадение"] == "Да").mean()
            if len(checked) else 0
        )

        col2.metric(
            "Точность статусов",
            f"{accuracy:.1%}" if len(checked) else "—"
        )

        col3.metric(
            "Среднее время, с",
            round(df["Время, с"].mean(), 2)
        )

        st.download_button(
            "Скачать результаты CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="results.csv",
            mime="text/csv"
        )
