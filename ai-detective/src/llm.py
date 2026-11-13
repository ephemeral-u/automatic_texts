
import time
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import ValidationError

from src.schemas import ModelAnswer


ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"



@lru_cache(maxsize=1)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    print(f"Устройство: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device
    )

    model.eval()
    return tokenizer, model



def ask_llm(question_id, question, context, available_claim_ids):
    tokenizer, model = load_model()

    system_prompt = (ROOT / "prompts" / "system.txt").read_text(
        encoding="utf-8"
    )

    user_prompt = (
        f"Вопрос ID: {question_id}\n"
        f"Вопрос: {question}\n\n"
        f"Доступные claim_id: {sorted(available_claim_ids)}\n\n"
        f"Контекст:\n{context}\n\n"
        "Верни только JSON с полями: question_id, status, "
        "answer, evidence_ids, counterevidence_ids, "
        "confidence, explanation."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    duration = time.perf_counter() - start

    input_tokens = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_tokens:]

    raw_answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    if raw_answer.startswith("```"):
        raw_answer = raw_answer.split("\n", 1)[1]
        raw_answer = raw_answer.rsplit("```", 1)[0].strip()

    try:
        answer = ModelAnswer.model_validate_json(raw_answer)
    except ValidationError as error:
        raise ValueError(
            f"Некорректный JSON от модели:\n{raw_answer}"
        ) from error

    if answer.question_id != question_id:
        raise ValueError("Неверный question_id")

    used_ids = set(
        answer.evidence_ids + answer.counterevidence_ids
    )

    if not used_ids.issubset(set(available_claim_ids)):
        raise ValueError("Модель указала недоступный claim_id")

    metrics = {
        "input_tokens": input_tokens,
        "output_tokens": len(generated_tokens),
        "duration_seconds": round(duration, 2)
    }

    return answer, metrics


if __name__ == "__main__":
    context = (
        "C20: Алексей Морозов вошёл в административную "
        "панель в 18:23:04.\n"
        "Источник D07: [18:23:04] ADMIN_LOGIN: "
        "user=alexey_morozov, status=success."
    )

    answer, metrics = ask_llm(
        question_id="Q01",
        question="Во сколько Алексей Морозов вошёл в административную панель?",
        context=context,
        available_claim_ids={"C20"}
    )

    print("Ответ модели:")
    print(answer.model_dump_json(indent=2))

    print("\nМетрики:")
    print(metrics)
