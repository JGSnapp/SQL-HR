FROM python:3.11-slim AS base

RUN groupadd -r app && useradd --no-log-init -r -g app app

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . ./
# каталогу конфигов Streamlit нужны права записи у пользователя app
RUN mkdir -p /app/.streamlit && chown -R app:app /app/.streamlit

USER app

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "chat.py"]
