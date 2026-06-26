import os
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent


def load_env(path=None):
    path = Path(path or BASE_DIR / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = os.path.expandvars(value.strip().strip('"').strip("'"))
        if key and key not in os.environ:
            os.environ[key] = value


class Config:
    def __init__(self):
        load_env()
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", os.getenv("AZURE_OPENAI_CHAT_API_VERSION", "2025-01-01-preview")).strip()
        self.azure_deployment = os.getenv("DDX_AZURE_DEPLOYMENT", os.getenv("AZURE_OPENAI_PLANNER_DEPLOYMENT", os.getenv("AZURE_OPENAI_DEPLOYMENT", ""))).strip()
        self.embedding_uri = os.getenv("EMBEDDING_URI", "").strip()
        self.embedding_auth = os.getenv("EMBEDDING_AUTH", "").strip()
        self.knowledge_db = ROOT_DIR / os.getenv("DDX_KNOWLEDGE_DB", os.getenv("KNOWLEDGE_DB_PATH", os.getenv("RASA_KNOWLEDGE_DB_PATH", "artifacts/knowledge/knowdge.db")))
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
        self.qdrant_collection = os.getenv("QDRANT_CHUNKS_COLLECTION", "clinical_chunks_v1").strip()
        self.storage_dir = ROOT_DIR / os.getenv("DDX_STORAGE_DIR", "ddx/storage")
        self.max_llm_tokens = int(os.getenv("DDX_MAX_LLM_TOKENS", "350"))
        key = re.sub(r"[^A-Za-z0-9]+", "_", self.azure_deployment).strip("_").upper()
        self.input_price_per_1m = float(os.getenv(f"{key}_INPUT", "0") or 0)
        self.cached_input_price_per_1m = float(os.getenv(f"{key}_CACHED_INPUT", "0") or 0)
        self.output_price_per_1m = float(os.getenv(f"{key}_OUTPUT", "0") or 0)
