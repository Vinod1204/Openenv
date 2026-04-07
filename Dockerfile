FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
        PIP_NO_CACHE_DIR=1 \
        HOST=0.0.0.0 \
        PORT=7860

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml models.py client.py __init__.py /app/
COPY server /app/server
COPY support_inbox /app/support_inbox
COPY inference.py /app/inference.py
COPY scripts /app/scripts

RUN pip install --upgrade pip && pip install -e .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3)"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

