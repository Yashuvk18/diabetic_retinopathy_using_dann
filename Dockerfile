FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

ENV PORT=8502
EXPOSE 8502

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8502} --workers 1 --threads 1 --timeout 180 --graceful-timeout 30 web_app:app"]
