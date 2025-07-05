# AI Text Summarizer - Backend API

Это бэкенд-часть проекта **AI Text Summarizer**. Сервис представляет собой API на FastAPI, которое принимает текст и возвращает его краткое содержание (саммари), сгенерированное с помощью T5-модели из библиотеки Hugging Face Transformers.

### ✨ Живые Демо

*   **Backend API (Hugging Face Spaces):** [https://huggingface.co/spaces/pahomm2116/summarizer-api](https://huggingface.co/spaces/pahomm2116/summarizer-api)
*   **Frontend (Vercel):** [https://summarizer-api-frontend.vercel.app/](https://summarizer-api-frontend.vercel.app/)

### 🚀 Технологический стек

*   **Фреймворк:** FastAPI
*   **Язык:** Python 3.11
*   **ML Библиотеки:** PyTorch, Hugging Face Transformers
*   **Контейнеризация:** Docker
*   **Менеджер версий:** Pyenv
*   **Деплой:** Hugging Face Spaces (с использованием Git LFS)

### 📋 Архитектура и возможности

*   **Модель:** `IlyaGusev/rut5_base_sum_gazeta` загружается из локальных файлов.
*   **Контейнеризация:** Приложение полностью упаковано в Docker-образ для портативности и воспроизводимости.
*   **Git LFS:** Для управления большим файлом модели (เกือบ 1 ГБ) используется Git Large File Storage.
*   **API:** Предоставляет эндпоинт `/summarize/` для получения саммари. Документация доступна через Swagger UI по эндпоинту `/docs`.

### ⚙️ Локальный запуск

1.  Установите [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2.  Клонируйте репозиторий (требуется Git LFS):
    ```bash
    git clone https://github.com/pahomm/summarizer-api.git
    ```
3.  Перейдите в папку проекта:
    ```bash
    cd summarizer-api
    ```
4.  Соберите Docker-образ:
    ```bash
    docker build -t summarizer-api .
    ```
5.  Запустите контейнер:
    ```bash
    docker run --rm -p 8000:7860 -it summarizer-api
    ```
6.  Откройте документацию API по адресу [http://localhost:8000/docs](http://localhost:8000/docs).

---
*Этот проект является частью полной end-to-end системы. Фронтенд-часть находится в [отдельном репозитории](https://github.com/pahomm/summarizer-api-frontend).*