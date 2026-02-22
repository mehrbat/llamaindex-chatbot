# LlamaIndex Chatbot

React + Vite frontend and FastAPI + LlamaIndex backend with per-thread memory.

## Prerequisites

- **Python 3.9 or 3.10** (3.11+ recommended; llama-index-core 0.12+ requires 3.10+)
- **Node.js 18+** and npm
- **OpenAI API key** (`OPENAI_API_KEY` env var)

## Backend (FastAPI + LlamaIndex)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python chatbot_backend.py
```

Backend runs at `http://localhost:8000`.

## Frontend (React + Vite)

```bash
npm install
npm run dev
```

Open the printed URL (usually `http://localhost:5173`).

## Usage

- Type a message and press **Send** to start a conversation.
- The first message creates a new thread; subsequent messages reuse it (memory per thread).
- Click **New Thread** to start a fresh conversation.

