from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pydantic import BaseModel

MODEL_PATH = "./models"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    tokenizer = None
    model = None


class SummarizationRequest(BaseModel):
    text: str
    min_length: int = 15
    max_length: int = 150
    num_beams: int = 4


def summarize(payload: SummarizationRequest) -> str:
    """
    Генерирует саммари для предоставленного текста с использованием T5 модели.

    Args:
        payload (SummarizationRequest): Объект Pydantic, содержащий данные для саммаризации.
            - text (str): Исходный текст, который необходимо сократить.
            - min_length (int): Минимальное количество токенов в сгенерированном саммари.
            - max_length (int): Максимальное количество токенов в сгенерированном саммари.
            - num_beams (int): Количество "лучей" для алгоритма Beam Search.

    Returns:
        str: Сгенерированное саммари.
    """
    if not tokenizer or not model:
        return "Ошибка: модель не была загружена."

    # Токенизация текста
    inputs = tokenizer(
        payload.text, return_tensors="pt", max_length=512, truncation=True
    )

    # Генерация саммари
    summary_ids = model.generate(
        inputs["input_ids"],
        min_length=payload.min_length,
        max_length=payload.max_length,
        num_beams=payload.num_beams,
        early_stopping=True,
        repetition_penalty=10.0,
        no_repeat_ngram_size=2,
    )

    # Декодирование и возврат результата
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
