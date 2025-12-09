# app/api.py
from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Request,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.templating import Jinja2Templates

from .config import BASE_DIR, KNOWLEDGE_DIR
from .vector_store import (
    get_vector_store,
    get_vector_store_info,
)
from .ingest import ingest_path, ingest_directory

app = FastAPI(title="Study RAG Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ---------- API models ----------


class StatusResponse(BaseModel):
    index_type: str
    embedding_dimension: int
    doc_count: int


class IngestPathRequest(BaseModel):
    path: str  # path under data/knowledge, or absolute inside container


class IngestResponse(BaseModel):
    chunks_added: int
    detail: str


class QueryRequest(BaseModel):
    question: str
    k: int = 5


class RetrievedChunk(BaseModel):
    source: str
    chunk_id: int
    text: str


class QueryResponse(BaseModel):
    answer: str
    retrieved: List[RetrievedChunk]


# ---------- Core RAG logic ----------


def run_rag_query(question: str, k: int = 5) -> QueryResponse:
    """
    Simple retrieval-only RAG:
    - searches FAISS
    - returns top-k chunks
    - 'answer' is just a stitched view of those chunks

    Project 2 (Copilot) will call this via /query and then hand off
    to an LLM for summarization / reasoning.
    """
    vs = get_vector_store()
    docs = vs.similarity_search(question, k=k)

    if not docs:
        return QueryResponse(
            answer="No documents in the index yet. Ingest something first.",
            retrieved=[],
        )

    context_snippets: List[str] = []
    retrieved: List[RetrievedChunk] = []

    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        chunk_id = int(d.metadata.get("chunk_id", 0))
        text = (d.page_content or "").strip()

        context_snippets.append(f"[{i}] {text}")
        retrieved.append(
            RetrievedChunk(
                source=src,
                chunk_id=chunk_id,
                text=text,
            )
        )

    answer = (
        "Top matching chunks from your library:\n\n"
        + "\n\n".join(context_snippets)
        + "\n\n(Next step: call an LLM service to turn this into a narrative answer.)"
    )

    return QueryResponse(answer=answer, retrieved=retrieved)


# ---------- JSON API endpoints ----------


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse)
def status():
    info = get_vector_store_info()
    return StatusResponse(**info)


@app.post("/ingest/path", response_model=IngestResponse)
def ingest_from_path(body: IngestPathRequest):
    raw_path = Path(body.path)

    if not raw_path.is_absolute():
        # Treat as relative to KNOWLEDGE_DIR
        target = KNOWLEDGE_DIR / raw_path
    else:
        target = raw_path

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {target}")

    if target.is_file():
        count = ingest_path(target)
        detail = f"Ingested file {target}"
    else:
        count = ingest_directory(target)
        detail = f"Ingested directory {target}"

    return IngestResponse(chunks_added=count, detail=detail)


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(file: UploadFile = File(...)):
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    dest = KNOWLEDGE_DIR / file.filename

    with dest.open("wb") as f:
        content = await file.read()
        f.write(content)

    count = ingest_path(dest)
    return IngestResponse(
        chunks_added=count,
        detail=f"Uploaded and ingested {dest}",
    )


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest):
    return run_rag_query(body.question, body.k)


# ---------- HTML UI endpoints ----------


@app.get("/", response_class=HTMLResponse)
def ui_home(request: Request):
    status = get_vector_store_info()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "status": status,
            "result": None,
            "question": "",
            "k": 5,
            "message": None,
        },
    )


@app.post("/ui/upload", response_class=HTMLResponse)
async def ui_upload(request: Request, file: UploadFile = File(...)):
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    dest = KNOWLEDGE_DIR / file.filename

    with dest.open("wb") as f:
        content = await file.read()
        f.write(content)

    chunks = ingest_path(dest)
    status = get_vector_store_info()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "status": status,
            "result": None,
            "question": "",
            "k": 5,
            "message": f"Uploaded {file.filename} as {chunks} chunks.",
        },
    )


@app.post("/ui/ingest-path", response_class=HTMLResponse)
async def ui_ingest_path(request: Request, path: str = Form(...)):
    raw_path = Path(path)

    if not raw_path.is_absolute():
        target = KNOWLEDGE_DIR / raw_path
    else:
        target = raw_path

    if not target.exists():
        status = get_vector_store_info()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "status": status,
                "result": None,
                "question": "",
                "k": 5,
                "message": f"Path not found: {target}",
            },
        )

    if target.is_file():
        count = ingest_path(target)
        detail = f"Ingested file {target}"
    else:
        count = ingest_directory(target)
        detail = f"Ingested directory {target}"

    status = get_vector_store_info()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "status": status,
            "result": None,
            "question": "",
            "k": 5,
            "message": f"{detail} ({count} chunks).",
        },
    )


@app.post("/ui/query", response_class=HTMLResponse)
async def ui_query(
    request: Request,
    question: str = Form(...),
    k: int = Form(5),
):
    status = get_vector_store_info()
    result = run_rag_query(question, k)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "status": status,
            "result": result,
            "question": question,
            "k": k,
            "message": None,
        },
    )
