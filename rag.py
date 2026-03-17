"""
Local RAG System — Ollama + LangChain + ChromaDB
Usage:
  python rag.py --index          → index all PDFs in ./docs/
  python rag.py --query "..."    → ask a question
"""

import os
import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Configuration ────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL     = "nomic-embed-text"
LLM_MODEL       = "llama3"
DOCS_DIR        = "./docs"
CHROMA_DIR      = "./chroma_db"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
TOP_K           = 3

PROMPT_TEMPLATE = """
You are an assistant that answers questions ONLY based on the provided documents.
If the answer is not in the context, say exactly:
"I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:
"""

# ── Helpers ──────────────────────────────────────────────────────────
def get_embeddings():
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )

def get_llm():
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0
    )

def get_vectorstore(embeddings, create=False, chunks=None):
    if create and chunks:
        print(f"  Creating ChromaDB with {len(chunks)} chunks...")
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
    print("  Loading existing ChromaDB...")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source','?')} | Page: {d.metadata.get('page','?')}]\n{d.page_content}"
        for d in docs
    )

# ── Index documents ──────────────────────────────────────────────────
def index_documents():
    print("\n📄 Loading PDFs from ./docs/ ...")

    if not Path(DOCS_DIR).exists():
        print("❌ ./docs/ folder not found. Please create it and add PDFs.")
        return

    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    if not documents:
        print("❌ No PDF files found in ./docs/")
        print("   → Copy your PDF courses into the docs/ folder first.")
        return

    print(f"✅ Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")

    embeddings = get_embeddings()
    print("🔄 Generating embeddings — this may take a few minutes...")

    vectorstore = get_vectorstore(embeddings, create=True, chunks=chunks)
    print(f"✅ Stored {vectorstore._collection.count()} chunks in ChromaDB")
    print("🎉 Done! You can now run: python rag.py --query \"your question\"")

# ── Query ────────────────────────────────────────────────────────────
def query(question: str):
    if not Path(CHROMA_DIR).exists():
        print("❌ No indexed documents found.")
        print("   → Run first: python rag.py --index")
        return

    print(f"\n❓ Question: {question}")
    print("🔄 Searching documents and generating answer...\n")

    embeddings  = get_embeddings()
    llm         = get_llm()
    vectorstore = get_vectorstore(embeddings)
    retriever   = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    print("🤖 Answer:")
    print("─" * 50)
    print(answer)
    print("─" * 50)
    return answer

# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local RAG System")
    parser.add_argument("--index", action="store_true", help="Index PDFs from ./docs/")
    parser.add_argument("--query", type=str, help="Ask a question")
    args = parser.parse_args()

    if args.index:
        index_documents()
    elif args.query:
        query(args.query)
    else:
        print("Usage:")
        print("  python rag.py --index")
        print("  python rag.py --query \"What is gradient descent?\"")