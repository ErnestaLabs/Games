FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY trading_games/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all required code
COPY trading_games/ ./trading_games/
COPY polymarket/ ./polymarket/

# Check geoblock before starting
CMD ["python", "-m", "trading_games.agent_runner"]
