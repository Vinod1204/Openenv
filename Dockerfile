FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml /app/
COPY support_inbox /app/support_inbox
COPY inference.py /app/inference.py
COPY scripts /app/scripts

RUN pip install --upgrade pip && pip install -e .

EXPOSE 7860

CMD ["uvicorn", "support_inbox.server:app", "--host", "0.0.0.0", "--port", "7860"]
