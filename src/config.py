# src/config.py
from pathlib import Path
from typing import Final, Literal

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV: Final = ROOT / "input" / "foerderkatalog_export.csv"
DATA_DIR: Final = ROOT / "data"
EMBEDDINGS_FILE: Final = DATA_DIR / "project_embeddings.npy"

# FAISS Index Files - jetzt provider-spezifisch
FAISS_INDEX_FILE_OLLAMA: Final = DATA_DIR / "vector.index"
FAISS_INDEX_FILE_HF: Final = DATA_DIR / "vector_hf.index"

# Embedding Map Files - provider-spezifisch
EMBED_MAP_FILE_OLLAMA: Final = DATA_DIR / "embeddings_map.json"
EMBED_MAP_FILE_HF: Final = DATA_DIR / "embeddings_map_hf.json"

# Progress Files - provider-spezifisch
PROGRESS_FILE_OLLAMA: Final = DATA_DIR / "indexing_progress.json"
PROGRESS_FILE_HF: Final = DATA_DIR / "indexing_progress_hf.json"

LOG_DIR: Final = ROOT / "logs"

# Embedding Provider Configuration
EmbeddingProvider = Literal["ollama", "huggingface"]
DEFAULT_EMBEDDING_PROVIDER: Final[EmbeddingProvider] = "ollama"

# Ollama Configuration
OLLAMA_EMBED_MODEL: Final = "nomic-embed-text"

# HuggingFace Configuration
HF_EMBED_MODEL_DEFAULT: Final = "sentence-transformers/all-mpnet-base-v2"
HF_EMBED_MODEL_ALTERNATIVES: Final = [
    "intfloat/e5-small-v2",  # 384 dim, schnell
    "sentence-transformers/all-MiniLM-L6-v2",  # 384 dim, sehr schnell
    "intfloat/e5-base-v2",  # 768 dim, besser
    "sentence-transformers/all-mpnet-base-v2",  # 768 dim, sehr gut
]

# LLM defaults
LLM_DEFAULT_MODEL: Final = "moonshotai/kimi-k2-instruct-0905"

# Limits
TOP_K_DEFAULT = 50
MAX_DOCS_FOR_LLM = 30


def get_index_files(provider: EmbeddingProvider) -> tuple[Path, Path, Path]:
    """Gibt die korrekten Index-Dateipfade für einen Provider zurück.

    Args:
        provider: "ollama" oder "huggingface"

    Returns:
        Tuple[Path, Path, Path]: (index_file, map_file, progress_file)
    """
    if provider == "ollama":
        return (FAISS_INDEX_FILE_OLLAMA, EMBED_MAP_FILE_OLLAMA, PROGRESS_FILE_OLLAMA)
    elif provider == "huggingface":
        return (FAISS_INDEX_FILE_HF, EMBED_MAP_FILE_HF, PROGRESS_FILE_HF)
    else:
        raise ValueError(f"Unbekannter Provider: {provider}")
