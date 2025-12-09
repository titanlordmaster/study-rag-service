FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch that satisfies the HF / transformers CVE guard
# Note the "+cpu" and the CPU wheel index.
RUN pip install --no-cache-dir \
      torch==2.6.0+cpu \
      --index-url https://download.pytorch.org/whl/cpu

# Now install the rest of your deps (no torch here!)
RUN pip install --no-cache-dir -r requirements.txt

# Hide host GPUs from inside the container so nothing even tries to use them
ENV CUDA_VISIBLE_DEVICES=""

COPY app ./app
COPY data ./data
COPY templates ./templates
COPY static ./static

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
