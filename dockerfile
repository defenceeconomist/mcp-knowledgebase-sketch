# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# (Optional but handy) system certs + curl for debugging/healthchecks
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000

CMD ["python", "mcp_app.py"]
