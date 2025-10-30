# src/config.py
from pathlib import Path
from typing import Final


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV: Final = ROOT / "input" / "foerderkatalog_export.csv"
DATA_DIR: Final = ROOT / "data"
EMBEDDINGS_FILE: Final = DATA_DIR / "project_embeddings.npy"
FAISS_INDEX_FILE: Final = DATA_DIR / "vector.index"
EMBED_MAP_FILE: Final = DATA_DIR / "embeddings_map.json"
LOG_DIR: Final = ROOT / "logs"


# Ollama/LLM defaults
OLLAMA_EMBED_MODEL: Final = "nomic-embed-text"
LLM_DEFAULT_MODEL: Final = "moonshotai/kimi-k2-instruct-0905"


# Limits
TOP_K_DEFAULT = 50
MAX_DOCS_FOR_LLM = 30
