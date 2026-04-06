FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY trading_games/ ./trading_games/
COPY polymarket/ ./polymarket/

CMD ["python", "-m", "trading_games.agent_runner"]
