FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml README.md /app/
COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health',timeout=3)"

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
