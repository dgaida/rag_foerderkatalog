# src/embeddings/faiss_store.py
import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from ..config import EmbeddingProvider, get_index_files

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class FaissStore:
    """Minimaler FAISS-Wrapper zum Persistieren von Embeddings.

    Funktionen:
    - Laden / Speichern Index (provider-spezifisch)
    - Hinzufügen von Embeddings (einzeln)
    - Suche nach k nächsten Nachbarn

    Attributes:
        provider: Embedding-Provider ("ollama" oder "huggingface")
        index_file: Pfad zur Index-Datei
        map_file: Pfad zur Mapping-Datei
        dim: Embedding-Dimension
        index: FAISS-Index
        id_map: Mapping von FAISS-IDs zu Document-IDs
    """

    def __init__(self, dim: int | None = None, provider: EmbeddingProvider = "ollama") -> None:
        """Initialisiert den FAISS-Store.

        Args:
            dim: Embedding-Dimension (optional)
            provider: "ollama" oder "huggingface"
        """
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.provider = provider
        self.dim = dim
        self.index = None
        self.id_map: dict[str, str] = {}

        # Hole provider-spezifische Dateipfade
        self.index_file, self.map_file, _ = get_index_files(provider)

        # Versuche Index zu laden
        if self.map_file.exists() and self.index_file.exists():
            try:
                self.id_map = json.loads(self.map_file.read_text(encoding="utf-8"))
                self.index = faiss.read_index(str(self.index_file))
                self.dim = self.index.d
            except Exception:
                self.index = None
                self.id_map = {}

    def _init_index(self, dim: int) -> None:
        """Initialisiert einen neuen FAISS-Index.

        Args:
            dim: Embedding-Dimension
        """
        self.dim = dim
        # Verwende Inner Product (cosine-normalisierte Vektoren)
        self.index = faiss.IndexFlatIP(dim)

    def clear(self) -> None:
        """Löscht den Index und alle Mappings."""
        self.index = None
        self.id_map = {}
        if self.index_file.exists():
            self.index_file.unlink()
        if self.map_file.exists():
            self.map_file.unlink()

    def add(self, vector: List[float], doc_id: str, persist_now: bool = True) -> None:
        """Fügt einen Embedding-Vektor zum Index hinzu.

        Args:
            vector: Embedding-Vektor als Liste von Floats
            doc_id: Eindeutige Document-ID
            persist_now: Wenn True, wird sofort persistiert

        Raises:
            ValueError: Wenn Vektor leer ist
        """
        if not vector or len(vector) == 0:
            raise ValueError(f"Leerer Embedding-Vektor für doc_id={doc_id}")

        vec = np.array(vector, dtype="float32").reshape(1, -1)
        if self.index is None:
            self._init_index(vec.shape[1])
        faiss.normalize_L2(vec)
        self.index.add(vec)
        new_id = self.index.ntotal - 1
        self.id_map[str(new_id)] = doc_id
        if persist_now:
            self.persist()

    def search(self, vector: List[float], k: int = 5) -> List[Tuple[float, str]]:
        """Sucht die k nächsten Nachbarn zu einem gegebenen Vektor.

        Führt eine Cosine-Similarity-Suche im FAISS-Index durch und gibt
        die k ähnlichsten Dokumente zurück, sortiert nach Relevanz.

        Args:
            vector: Embedding-Vektor als Liste von Floats.
            k: Anzahl der zurückzugebenden nächsten Nachbarn. Defaults to 5.

        Returns:
            List[Tuple[float, str]]: Liste von Tupeln (Similarity-Score, doc_id),
                sortiert nach absteigendem Score. Leere Liste wenn Index leer ist.

        Example:
            >>> store = FaissStore(provider="ollama")
            >>> store.add([0.1] * 768, doc_id="doc_1")
            >>> results = store.search([0.1] * 768, k=5)
            >>> results[0]
            (0.9999, 'doc_1')

        Note:
            Der Input-Vektor wird automatisch L2-normalisiert für Cosine-Similarity.
            Scores liegen zwischen -1 und 1, wobei höhere Werte bessere Matches bedeuten.
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        vec = np.array(vector, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        distances, indices = self.index.search(vec, k)
        results: List[Tuple[float, str]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            fid = str(idx)
            filepath = self.id_map.get(fid)
            if filepath:
                results.append((float(dist), filepath))
        return results

    def persist(self) -> None:
        """Persistiert Index und Mapping auf Disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
        with open(self.map_file, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, ensure_ascii=False, indent=2)

    def get_info(self) -> dict:
        """Gibt Informationen über den Index zurück.

        Returns:
            dict: Dictionary mit Index-Informationen
        """
        return {
            "provider": self.provider,
            "index_file": str(self.index_file),
            "map_file": str(self.map_file),
            "dimension": self.dim,
            "total_vectors": self.index.ntotal if self.index else 0,
            "exists": self.index_file.exists(),
        }
