"""
RAG REST API — FastAPI server
Endpoints:
  GET  /           → health check
  POST /upload     → upload & index a PDF
  POST /query      → ask a question (with optional document filter)
  GET  /documents  → list indexed documents
  DELETE /reset    → clear the database
"""

import os
import shutil
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Configuration ────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL     = "nomic-embed-text"
LLM_MODEL       = "llama3.2:3b"
DOCS_DIR        = "./docs"
CHROMA_DIR      = "./chroma_db"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
TOP_K           = 6

PROMPT_TEMPLATE = """
You are an assistant that answers questions based ONLY on the provided documents.
The documents may be in French or English. Always answer in the same language as the question.
If the answer is not in the context, say exactly:
"I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:
"""

# ── Shared model instances ────────────────────────────────────────────
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
llm        = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

# ── Lifespan — warms up LLM at startup ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pre-load the LLM into GPU memory
    print("🔥 Warming up LLM into GPU memory...")
    try:
        llm.invoke("hello")
        print("✅ LLM is ready — GPU warmed up!")
    except Exception as e:
        print(f"⚠️  Warmup failed (will load on first query): {e}")
    yield
    # Shutdown
    print("👋 Shutting down RAG API...")

# ── App setup ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Local RAG API",
    description="Ask questions about your PDF documents using a local LLM",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source','?')} | Page: {d.metadata.get('page','?')}]\n{d.page_content}"
        for d in docs
    )

def retrieve_docs(question: str, document_filter: Optional[str] = None):
    """
    Retrieve relevant chunks.
    If document_filter is provided, filter results manually by filename.
    """
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    if document_filter:
        # Fetch a large pool then filter manually by filename
        candidates = vectorstore.similarity_search(question, k=50)
        filtered = [
            doc for doc in candidates
            if document_filter.lower() in
               Path(doc.metadata.get("source", "")).name.lower()
        ]
        print(f"  Filter '{document_filter}': {len(filtered)} matches from {len(candidates)} candidates")
        return filtered[:TOP_K]
    else:
        return vectorstore.similarity_search(question, k=TOP_K)


def build_answer(question: str, docs: list) -> str:
    """Build a prompt from retrieved docs and invoke the LLM."""
    if not docs:
        return "I don't know based on the provided documents."

    context = format_docs(docs)
    prompt  = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain   = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

# ── Request / Response models ─────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    document: Optional[str] = None

class QuestionResponse(BaseModel):
    question: str
    answer: str
    document: Optional[str] = None
    sources: list = []

# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "RAG API is ready",
        "model": LLM_MODEL,
        "docs_url": "/docs"
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and index it into ChromaDB"""

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    Path(DOCS_DIR).mkdir(exist_ok=True)
    dest = Path(DOCS_DIR) / file.filename

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"📄 Saved: {dest}")

    loader    = PyPDFLoader(str(dest))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print(f"✅ Indexed {len(chunks)} chunks from {file.filename}")

    return {
        "message": f"Successfully indexed {file.filename}",
        "pages": len(documents),
        "chunks": len(chunks),
        "filename": file.filename
    }


@app.post("/query", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    """
    Ask a question.
    - document=null     → search across ALL indexed documents
    - document="x.pdf"  → search ONLY inside that specific file
    """

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not Path(CHROMA_DIR).exists():
        raise HTTPException(
            status_code=404,
            detail="No documents indexed yet. Upload a PDF first via POST /upload"
        )

    scope = f"[{request.document}]" if request.document else "[all docs]"
    print(f"❓ Query {scope}: {request.question}")

    docs   = retrieve_docs(request.question, document_filter=request.document)
    answer = build_answer(request.question, docs)

    sources = list({
        Path(d.metadata.get("source", "")).name
        for d in docs
    })

    print(f"🤖 Answer from {sources}: {answer[:80]}...")

    return QuestionResponse(
        question=request.question,
        answer=answer,
        document=request.document,
        sources=sources
    )


@app.get("/documents")
def list_documents():
    """List all indexed PDF files"""
    docs_path = Path(DOCS_DIR)
    if not docs_path.exists():
        return {"documents": [], "count": 0}

    pdfs = list(docs_path.glob("*.pdf"))
    return {
        "documents": [p.name for p in pdfs],
        "count": len(pdfs)
    }


@app.delete("/reset")
def reset_database():
    """Delete all indexed documents and clear ChromaDB"""
    deleted = []

    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)
        deleted.append("chroma_db")

    if Path(DOCS_DIR).exists():
        shutil.rmtree(DOCS_DIR)
        deleted.append("docs")

    return {"message": "Database cleared", "deleted": deleted}

from fastapi.responses import FileResponse

@app.post("/video/generate")
def generate_video_endpoint(course: str, chapter: str):
    from video import create_video

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    docs    = vectorstore.similarity_search("introduction overview summary", k=15)
    context = "\n\n".join(d.page_content for d in docs)

    video_path = create_video(
        chapter_name=f"{course}_{chapter}",
        context=context,
        base_url=OLLAMA_BASE_URL
    )

    if not video_path:
        raise HTTPException(status_code=500, detail="Video generation failed.")

    return {"message": "Video ready", "path": video_path}


@app.get("/video/{course}/{chapter}")
def download_video(course: str, chapter: str):
    safe_name  = f"{course}_{chapter}".replace(" ", "_")
    video_path = Path("./videos") / f"{safe_name}_course.mp4"

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found. Generate it first.")

    return FileResponse(
        str(video_path),
        media_type="video/mp4",
        filename=f"{safe_name}.mp4"
    )