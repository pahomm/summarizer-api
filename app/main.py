from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from .model import summarize, SummarizationRequest, model, tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Запуск приложения...")
    if model is None or tokenizer is None:
        raise RuntimeError(
            "Не удалось загрузить ML модель или токенизатор. Приложение не может стартовать."
        )

    yield

    print("Остановка приложения...")


description = """
SUMMARIZER API поможет вам получить краткое содержание длинного текста.
Вставьте ваш текст и получите его саммари. 🚀
"""

app = FastAPI(
    title="Summarizer API",
    description=description,
    version="0.1.0",
    lifespan=lifespan,
)


class SummarizationResponse(BaseModel):
    summary: str


@app.post("/summarize/", response_model=SummarizationResponse)
async def get_summary(payload: SummarizationRequest):
    """
    Принимает текст и параметры для генерации саммари.

    - **text**: Исходный текст для саммаризации (обязательный параметр).
    - **min_length**: Минимальная длина итогового саммари в токенах (по умолчанию: 40).
    - **max_length**: Максимальная длина итогового саммари в токенах (по умолчанию: 150).
    - **num_beams**: Количество лучей для поиска (beam search). Увеличивает качество, но замедляет генерацию (по умолчанию: 4).
    """
    try:
        summary_text = summarize(payload)
        return {"summary": summary_text}
    except Exception as e:
        print(f"Произошла ошибка при саммаризации: {e}")
        raise HTTPException(
            status_code=500, detail="Внутренняя ошибка сервера при обработке текста."
        )


@app.get("/")
async def root():
    return {"message": "Welcome to the Summarizer API. Go to /docs for documentation."}
