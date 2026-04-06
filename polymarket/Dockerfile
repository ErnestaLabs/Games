FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY polymarket/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the polymarket module and its dependencies
COPY polymarket/ ./polymarket/
COPY polymarket/__init__.py ./polymarket/__init__.py

# Data directory for predictions.jsonl
RUN mkdir -p /app/polymarket/data

# Run both bot + resolution_checker in parallel via a simple launcher
COPY polymarket/run_all.py ./run_all.py

CMD ["python", "run_all.py"]
