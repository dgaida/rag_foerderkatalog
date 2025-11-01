# src/embeddings/faiss_store.py
from pathlib import Path
import numpy as np
import faiss
import json
from typing import List, Tuple

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INDEX_FILE = DATA_DIR / "vector.index"
MAP_FILE = DATA_DIR / "embeddings_map.json"


class FaissStore:
    """Minimaler FAISS-Wrapper zum Persistieren von Embeddings.

    Funktionen:
    - Laden / Speichern Index
    - Hinzufügen von Embeddings (einzeln)
    - Suche nach k nächsten Nachbarn
    """

    def __init__(self, dim: int | None = None) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.index = None
        self.id_map: dict[str, str] = {}
        if MAP_FILE.exists() and INDEX_FILE.exists():
            try:
                self.id_map = json.loads(MAP_FILE.read_text(encoding="utf-8"))
                self.index = faiss.read_index(str(INDEX_FILE))
                self.dim = self.index.d
            except Exception:
                self.index = None
                self.id_map = {}

    def _init_index(self, dim: int) -> None:
        self.dim = dim
        # Verwende Inner Product (cosine-normalisierte Vektoren)
        self.index = faiss.IndexFlatIP(dim)

    def clear(self) -> None:
        self.index = None
        self.id_map = {}
        if INDEX_FILE.exists():
            INDEX_FILE.unlink()
        if MAP_FILE.exists():
            MAP_FILE.unlink()

    def add(self, vector: List[float], doc_id: str, persist_now: bool = True) -> None:
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
            >>> store = FaissStore()
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
        if self.index is not None:
            faiss.write_index(self.index, str(INDEX_FILE))
        with open(MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, ensure_ascii=False, indent=2)
