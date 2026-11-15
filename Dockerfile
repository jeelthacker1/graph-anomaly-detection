FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml constraints-serving.txt ./
COPY fraudgraph ./fraudgraph
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -c constraints-serving.txt .
COPY live_app.py ./
COPY runs/payments ./runs/payments
RUN useradd --uid 10001 --create-home fraudgraph && mkdir -p /app/state && chown -R fraudgraph:fraudgraph /app/state
USER 10001
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FRAUD_DATABASE=/app/state/payments.sqlite FRAUD_MODEL_DIR=/app/runs/payments
EXPOSE 8000
CMD ["uvicorn","fraudgraph.payments.api:create_app","--factory","--host","0.0.0.0","--port","8000","--workers","1","--limit-concurrency","64"]
