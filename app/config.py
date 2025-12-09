# app/config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Where raw docs live (inside the container)
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"

# Where FAISS index & metadata live
VECTOR_DIR = BASE_DIR / "data" / "vector_store"

# Embeddings (HF, not Ollama)
# Default: BAAI/bge-m3 on GPU. You can override via env vars.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda")  # "cuda" or "cpu"
