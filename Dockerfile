
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/app

WORKDIR /app

# Create a non-root application user
RUN useradd --create-home --home-dir /home/app app

# Install Python dependencies
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install "temporalio>=1.10" "uvicorn[standard]"

# Copy application files
COPY --chown=app:app app/ ./app/
COPY --chown=app:app static/ ./static/

# Create the application settings directory
RUN mkdir -p /home/app/.scribble2structure \
    && chown -R app:app /app /home/app/.scribble2structure

USER app

EXPOSE 8090

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
