# 🧠 DocMind — Local RAG System

A fully **local**, **private**, and **GPU-accelerated** RAG system with YouTube-style video generation, Tunisian dialect support, and AI quiz generation.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ Features

- 📄 **PDF Upload & Indexing** — drag & drop PDF documents
- 🔍 **Semantic Search** — vector embeddings with ChromaDB
- 🧠 **Local LLM** — 100% private, no cloud, no API key
- ⚡ **GPU Accelerated** — NVIDIA GPU support via Ollama
- 🗣️ **Tunisian Dialect (Derja)** — understands Arabic dialect in Latin chars
- 🎬 **YouTube-Style Video** — animated educational videos from documents
- 📝 **AI Quiz Generator** — auto-generates quizzes with 80% progression gate
- 🗂️ **Chapter-based Structure** — courses divided into chapters
- 🐳 **Fully Dockerized** — one command to run everything

---

## 🏗️ Architecture

```
Browser (Next.js — port 3000)
         │
         ▼
FastAPI REST API (Docker — port 8000)
         │
         ├── ChromaDB      → Vector database (per chapter)
         ├── quiz.py       → AI Quiz generator
         ├── derja.py      → Tunisian dialect support
         └── video.py      → YouTube-style video pipeline
                  │
                  └── Ollama (port 11434)
                        ├── llama3.2:3b   → answers + scripts
                        └── nomic-embed-text → embeddings
```

---

## 🎬 Video Pipeline

```
PDF Document
     ↓
Extract key content (ChromaDB)
     ↓
Generate script (5 slides) ← LLM
     ↓
Animated frames (Pillow + NumPy)
  - Gradient background with wave animation
  - Progressive bullet point reveal
  - 5 color themes
  - Progress bar + slide counter
     ↓
Natural voice narration (edge-tts Microsoft Neural)
     ↓
Final MP4 video (MoviePy)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | llama3.2:3b via Ollama |
| Embeddings | nomic-embed-text |
| Vector DB | ChromaDB |
| Backend API | FastAPI + Python 3.12 |
| RAG Framework | LangChain |
| Video | MoviePy + Pillow + NumPy |
| TTS | edge-tts (Microsoft Neural) |
| Container | Docker + Docker Compose |

---

## 📋 Prerequisites

- [Ollama](https://ollama.com/download) installed and running
- [Docker Desktop](https://www.docker.com/products/docker-desktop) running
- [Python 3.12+](https://www.python.org/) for local scripts
- NVIDIA GPU recommended (works on CPU too)

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/selmirabii741/rag-system.git
cd rag-system
```

### 2. Pull Ollama models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 3. Start Ollama

```bash
ollama serve
```

### 4. Start the API with Docker

```bash
docker-compose up --build
```

### 5. Upload a PDF

```python
# upload_test.py
import requests, os

pdf_path = r"your_document.pdf"
with open(pdf_path, "rb") as f:
    r = requests.post(
        "http://localhost:8000/upload",
        files={"file": (os.path.basename(pdf_path), f, "application/pdf")}
    )
print(r.json())
```

### 6. Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quest-ce que la conteneurisation?"}'
```

### 7. Generate a YouTube-style video

```bash
curl -X POST "http://localhost:8000/video/generate?course=Docker&chapter=chapter1"
```

---

## 📁 Project Structure

```
rag-system/
├── api.py                  ← FastAPI REST server
├── rag.py                  ← RAG pipeline (CLI)
├── quiz.py                 ← AI Quiz generator
├── derja.py                ← Tunisian dialect support
├── video.py                ← YouTube-style video pipeline
├── upload_test.py          ← PDF upload script
├── requirements.txt        ← Python dependencies
├── Dockerfile              ← Docker image
├── docker-compose.yml      ← Docker orchestration
├── .gitignore
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload & index a PDF |
| `POST` | `/query` | Ask a question (Derja supported) |
| `GET` | `/documents` | List indexed documents |
| `POST` | `/video/generate` | Generate YouTube-style video |
| `GET` | `/video/{course}/{chapter}` | Download generated video |
| `DELETE` | `/reset` | Clear all data |

### Example: Ask in Tunisian Dialect

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "chnowa Docker?"}'
```

Response:
```json
{
  "question": "chnowa Docker?",
  "answer": "Docker هو أداة للـ containerisation...",
  "is_derja": true,
  "translated_query": "ما هو Docker"
}
```

---

## ⚙️ Configuration

Edit `api.py` to customize:

```python
LLM_MODEL    = "llama3.2:3b"      # Change LLM model
EMBED_MODEL  = "nomic-embed-text"  # Change embedding model
CHUNK_SIZE   = 500                 # Characters per chunk
CHUNK_OVERLAP = 50                 # Overlap between chunks
TOP_K        = 6                   # Chunks to retrieve
```

---

## 📊 Performance

Tested on **NVIDIA GeForce RTX 2050 (4GB VRAM)**:

| Metric | Value |
|--------|-------|
| Model | llama3.2:3b (2.8 GB) |
| GPU Usage | 100% GPU |
| Query response | ~3-4 seconds |
| Video generation | ~3-5 minutes |
| Embedding model | 100% GPU |

---

## 🐳 Docker Commands

```bash
# Start
docker-compose up

# Rebuild after code changes
docker-compose up --build

# Stop
docker-compose down

# View logs
docker logs rag-api -f

# Full rebuild (no cache)
docker-compose build --no-cache
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Connection refused` port 8000 | Run `docker-compose up` |
| `Connection refused` Ollama | Run `ollama serve` |
| Slow responses (>20s) | GPU not fully used — check `ollama ps` |
| Video generation fails | Check `docker logs rag-api --tail 50` |
| "I don't know" answers | Ask in document's language (FR/EN) |
| JSON parse error in video | Context too long — already fixed with 1500 char limit |

---

## 🗣️ Tunisian Dialect Support

The system detects and understands Tunisian Derja:

| Derja | Meaning |
|-------|---------|
| `chnowa` | What is |
| `kifeh` | How |
| `3lech` | Why |
| `win` | Where |
| `mta3` | Of / belonging to |
| `barcha` | A lot |

Add more words in `derja.py` → `DERJA_MAP`.

---

## 📄 License

MIT License — free to use, modify and distribute.

---

## 👨‍💻 Author

**selmirabii741** — Built with ❤️ using Ollama, LangChain, FastAPI, MoviePy and Next.js
