FROM python:3.11-slim AS base

RUN groupadd -r app && useradd --no-log-init -r -g app app

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . ./

USER app

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
