# The agent under test of the Harbor example: an agno AgentOS app on port 7777.
# Build from integrations/harbor: docker build -t simulhausen-agentos-demo examples/agentos
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1

RUN uv pip install --system --no-cache "agno[os]==3.1.1" "openai==3.24.0"

WORKDIR /app
COPY app.py .

EXPOSE 7777
CMD ["python", "app.py"]
