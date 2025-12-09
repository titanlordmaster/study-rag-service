from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# Base paths
VECTOR_DIR = Path("data/vector_store")
INDEX_DIR = VECTOR_DIR  # FAISS.save_local / load_local use a directory

# Embedding model config
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
DEFAULT_EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

# We are intentionally CPU-only inside this container
EMBED_DEVICE = "cpu"

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vector_store: Optional[FAISS] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Singleton HF embeddings client (BAAI/bge-m3 by default).

    Critical: force device='cpu' so we never touch broken CUDA kernels
    in this container. Your GPUs are sm_61/sm_50 and the torch wheel
    inside the image only supports sm_70+.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=DEFAULT_EMBED_MODEL,
            model_kwargs={"device": EMBED_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _load_vector_store() -> Optional[FAISS]:
    if not VECTOR_DIR.exists():
        return None

    try:
        vs = FAISS.load_local(
            str(VECTOR_DIR),
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        print("[vector_store] Loaded existing FAISS index from disk.")
        return vs
    except Exception as e:
        print(
            "[vector_store] Failed to load existing index, recreating. "
            f"Reason: {e}"
        )
        return None


def _create_empty_vector_store() -> FAISS:
    """
    Create a brand-new FAISS index.

    We *try* to probe the true embedding dimension by running a
    single CPU embedding; if that fails, we fall back to the
    configured DEFAULT_EMBED_DIM.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    dim = DEFAULT_EMBED_DIM

    try:
        probe_vec = _get_embeddings().embed_query("dimension probe")
        dim = len(probe_vec)
        print(f"[vector_store] Probed embedding dim from HF model: {dim}")
    except Exception as e:
        print(
            "[vector_store] WARNING: could not probe embedding dimension, "
            f"defaulting to {dim}. Error: {e}"
        )

    index = faiss.IndexFlatL2(dim)
    vs = FAISS(
        embedding_function=_get_embeddings(),
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )
    print(
        f"[vector_store] Created new FAISS index with dim={dim} "
        f"for model={DEFAULT_EMBED_MODEL}."
    )
    return vs


def get_vector_store() -> FAISS:
    global _vector_store
    if _vector_store is None:
        vs = _load_vector_store()
        if vs is None:
            vs = _create_empty_vector_store()
        _vector_store = vs
    return _vector_store


def save_vector_store(vs: Optional[FAISS] = None) -> None:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    vs = vs or get_vector_store()
    vs.save_local(str(VECTOR_DIR))


def add_documents_to_vector_store(
    vs: FAISS,
    documents: List[Document],
    persist: bool = True,
) -> FAISS:
    if not documents:
        return vs

    vs.add_documents(documents)
    print(f"[vector_store] Added {len(documents)} documents to FAISS.")

    if persist:
        save_vector_store(vs)

    return vs


def count_documents_in_vector_store(vs: Optional[FAISS] = None) -> int:
    vs = vs or get_vector_store()
    try:
        return int(vs.index.ntotal)  # type: ignore[attr-defined]
    except Exception:
        return 0


def retrieve_all_documents(vs: Optional[FAISS] = None) -> List[Document]:
    vs = vs or get_vector_store()
    # InMemoryDocstore stores data in `_dict`
    return list(vs.docstore._dict.values())  # type: ignore[attr-defined]


def get_vector_store_info(vs: Optional[FAISS] = None) -> Dict[str, Any]:
    vs = vs or get_vector_store()
    index_type = type(vs.index).__name__
    dim = getattr(vs.index, "d", None)
    doc_count = count_documents_in_vector_store(vs)
    return {
        "index_type": index_type,
        "embedding_dimension": dim,
        "doc_count": doc_count,
    }


def search_vector_store(
    vs: FAISS,
    query: str,
    k: int = 5,
):
    return vs.similarity_search(query, k=k)
