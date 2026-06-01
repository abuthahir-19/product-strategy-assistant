import os
from dotenv import load_dotenv

# Load .env from the project root (one level up from backend/)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_root, ".env"))

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://keygateway.arshnivlabs.com/v1")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ChromaDB persists inside the project root
CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR",
    os.path.join(_root, "chroma_db")
)

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_RETRIEVAL_DOCS: int = int(os.getenv("MAX_RETRIEVAL_DOCS", "5"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1200"))
BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")
