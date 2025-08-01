FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app

COPY ./models /app/models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]