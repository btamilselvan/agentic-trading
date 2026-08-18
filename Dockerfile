FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY alembic.ini ./
COPY alembic ./alembic
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "agentic_trading.main:app", "--host", "0.0.0.0", "--port", "8000"]
