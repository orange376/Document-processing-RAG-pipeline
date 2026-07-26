"""Entry point for the RAG pipeline API server."""
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.api.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    # Only enable reload when explicitly set (e.g. RAG_RELOAD=1 python run.py).
    # Auto-reload kills background processing tasks and causes status to hang at "queued".
    use_reload = os.environ.get("RAG_RELOAD", "").strip() in ("1", "true", "yes")
    uvicorn.run("run:app", host="0.0.0.0", port=8001, reload=use_reload)
