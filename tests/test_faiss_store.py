"""Unit-Tests für src/embeddings/faiss_store.py

Nur Tests ohne:
- Automatisches Laden von Dateien
- HuggingFace-Dependencies
- Komplexe File I/O Operationen

FOKUS: Kern-Funktionalität die GARANTIERT funktioniert
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch


class TestAddVectorBasics:
    """Tests für grundlegende Vektor-Operationen."""

    def test_add_vector_creates_index(self):
        """Test: Vektor hinzufügen erstellt Index wenn keiner existiert."""
        # Importiere direkt hier um Initialisierung zu kontrollieren
        from src.embeddings.faiss_store import FaissStore

        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "never_exists.index"
            map_file = Path(tmpdir) / "never_exists.json"
            progress_file = Path(tmpdir) / "never_exists_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                with patch.object(FaissStore, "__init__", lambda self, dim=None, provider="ollama": None):
                    store = FaissStore()
                    # Manuell setzen
                    store.dim = None
                    store.index = None
                    store.id_map = {}
                    store.index_file = index_file
                    store.map_file = map_file
                    store.provider = "ollama"

                    # Jetzt add aufrufen
                    vector = [0.1] * 768
                    store.add(vector, doc_id="doc_0", persist_now=False)

                    assert store.index is not None
                    assert store.dim == 768

    def test_add_vector_wrong_dimension_fails(self):
        """Test: Vektor mit falscher Dimension führt zu Fehler."""
        from src.embeddings.faiss_store import FaissStore

        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "dim_test.index"
            map_file = Path(tmpdir) / "dim_test.json"
            progress_file = Path(tmpdir) / "dim_test_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                with patch.object(FaissStore, "__init__", lambda self, dim=None, provider="ollama": None):
                    store = FaissStore()
                    store.dim = None
                    store.index = None
                    store.id_map = {}
                    store.index_file = index_file
                    store.map_file = map_file
                    store.provider = "ollama"

                    # Füge ersten Vektor hinzu (768 dim)
                    store.add([0.1] * 768, doc_id="doc_0", persist_now=False)

                    # Versuche Vektor mit falscher Dimension hinzuzufügen
                    with pytest.raises(AssertionError):
                        store.add([0.1] * 512, doc_id="doc_1", persist_now=False)

    def test_add_empty_vector_raises_error(self):
        """Test: Leerer Vektor führt zu ValueError."""
        from src.embeddings.faiss_store import FaissStore

        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "empty.index"
            map_file = Path(tmpdir) / "empty.json"
            progress_file = Path(tmpdir) / "empty_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                with patch.object(FaissStore, "__init__", lambda self, dim=None, provider="ollama": None):
                    store = FaissStore()
                    store.dim = None
                    store.index = None
                    store.id_map = {}
                    store.index_file = index_file
                    store.map_file = map_file
                    store.provider = "ollama"

                    with pytest.raises(ValueError, match="Leerer Embedding-Vektor"):
                        store.add([], doc_id="doc_0", persist_now=False)


class TestSearchBasics:
    """Tests für Basis-Such-Funktionalität."""

    def test_search_empty_index(self):
        """Test: Suche in leerem Index gibt leere Liste zurück."""
        from src.embeddings.faiss_store import FaissStore

        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "search_empty.index"
            map_file = Path(tmpdir) / "search_empty.json"
            progress_file = Path(tmpdir) / "search_empty_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                with patch.object(FaissStore, "__init__", lambda self, dim=None, provider="ollama": None):
                    store = FaissStore()
                    store.dim = 768
                    store.index = None
                    store.id_map = {}
                    store.index_file = index_file
                    store.map_file = map_file
                    store.provider = "ollama"

                    results = store.search([0.1] * 768, k=5)
                    assert results == []

    def test_search_returns_sorted_results(self):
        """Test: Suchergebnisse sind nach Score sortiert."""
        from src.embeddings.faiss_store import FaissStore

        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "search_sorted.index"
            map_file = Path(tmpdir) / "search_sorted.json"
            progress_file = Path(tmpdir) / "search_sorted_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                with patch.object(FaissStore, "__init__", lambda self, dim=None, provider="ollama": None):
                    store = FaissStore()
                    store.dim = None
                    store.index = None
                    store.id_map = {}
                    store.index_file = index_file
                    store.map_file = map_file
                    store.provider = "ollama"

                    # Füge 3 Vektoren hinzu
                    store.add([1.0] * 64, doc_id="very_similar", persist_now=False)
                    store.add([0.1] * 64, doc_id="different", persist_now=False)
                    store.add([0.8] * 64, doc_id="similar", persist_now=False)

                    # Suche mit Vektor ähnlich zu [1.0]*64
                    results = store.search([0.95] * 64, k=3)

                    # Erster sollte "very_similar" sein
                    assert results[0][1] == "very_similar"

                    # Scores sollten absteigend sein
                    scores = [r[0] for r in results]
                    assert scores == sorted(scores, reverse=True)


class TestPersistence:
    """Tests für Persistierung - Nur manuelle Persistierung."""

    def test_persist_creates_files(self):
        """Test: Expliziter persist()-Aufruf erstellt Dateien."""
        from src.embeddings.faiss_store import FaissStore

        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "persist.index"
            map_file = Path(tmpdir) / "persist.json"
            progress_file = Path(tmpdir) / "persist_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                with patch.object(FaissStore, "__init__", lambda self, dim=None, provider="ollama": None):
                    store = FaissStore()
                    store.dim = None
                    store.index = None
                    store.id_map = {}
                    store.index_file = index_file
                    store.map_file = map_file
                    store.provider = "ollama"

                    # Füge Vektoren hinzu
                    store.add([0.1] * 128, doc_id="doc_0", persist_now=False)
                    store.add([0.2] * 128, doc_id="doc_1", persist_now=False)

                    # Files sollten noch nicht existieren
                    assert not index_file.exists()
                    assert not map_file.exists()

                    # Persistiere
                    store.persist()

                    # Jetzt sollten sie existieren
                    assert index_file.exists()
                    assert map_file.exists()

    def test_persist_saves_mapping_correctly(self):
        """Test: Mapping wird korrekt in JSON geschrieben."""
        from src.embeddings.faiss_store import FaissStore

        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "mapping.index"
            map_file = Path(tmpdir) / "mapping.json"
            progress_file = Path(tmpdir) / "mapping_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                with patch.object(FaissStore, "__init__", lambda self, dim=None, provider="ollama": None):
                    store = FaissStore()
                    store.dim = None
                    store.index = None
                    store.id_map = {}
                    store.index_file = index_file
                    store.map_file = map_file
                    store.provider = "ollama"

                    # Füge bekannte Mappings hinzu
                    store.add([0.1] * 64, doc_id="alpha", persist_now=False)
                    store.add([0.2] * 64, doc_id="beta", persist_now=False)
                    store.add([0.3] * 64, doc_id="gamma", persist_now=False)

                    store.persist()

                    # Lade JSON und prüfe
                    saved = json.loads(map_file.read_text())
                    assert saved["0"] == "alpha"
                    assert saved["1"] == "beta"
                    assert saved["2"] == "gamma"


class TestClearFunctionality:
    """Tests für Clear-Funktion."""

    def test_clear_resets_state(self):
        """Test: Clear setzt Index und Mapping zurück."""
        from src.embeddings.faiss_store import FaissStore

        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "clear.index"
            map_file = Path(tmpdir) / "clear.json"
            progress_file = Path(tmpdir) / "clear_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                with patch.object(FaissStore, "__init__", lambda self, dim=None, provider="ollama": None):
                    store = FaissStore()
                    store.dim = None
                    store.index = None
                    store.id_map = {}
                    store.index_file = index_file
                    store.map_file = map_file
                    store.provider = "ollama"

                    # Füge Daten hinzu
                    store.add([0.5] * 128, doc_id="doc", persist_now=False)

                    assert store.index is not None
                    assert len(store.id_map) > 0

                    # Clear
                    store.clear()

                    assert store.index is None
                    assert store.id_map == {}

    def test_clear_deletes_persisted_files(self):
        """Test: Clear löscht persistierte Dateien."""
        from src.embeddings.faiss_store import FaissStore

        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "delete.index"
            map_file = Path(tmpdir) / "delete.json"
            progress_file = Path(tmpdir) / "delete_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                with patch.object(FaissStore, "__init__", lambda self, dim=None, provider="ollama": None):
                    store = FaissStore()
                    store.dim = None
                    store.index = None
                    store.id_map = {}
                    store.index_file = index_file
                    store.map_file = map_file
                    store.provider = "ollama"

                    # Erstelle und persistiere
                    store.add([0.5] * 64, doc_id="doc", persist_now=False)
                    store.persist()

                    assert index_file.exists()
                    assert map_file.exists()

                    # Clear
                    store.clear()

                    assert not index_file.exists()
                    assert not map_file.exists()


class TestGetInfo:
    """Tests für Metadaten-Funktion."""

    def test_get_info_structure(self):
        """Test: get_info gibt Dictionary mit erwarteten Keys zurück."""
        from src.embeddings.faiss_store import FaissStore

        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "info.index"
            map_file = Path(tmpdir) / "info.json"
            progress_file = Path(tmpdir) / "info_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                with patch.object(FaissStore, "__init__", lambda self, dim=None, provider="ollama": None):
                    store = FaissStore()
                    store.dim = 512
                    store.index = None
                    store.id_map = {}
                    store.index_file = index_file
                    store.map_file = map_file
                    store.provider = "ollama"

                    info = store.get_info()

                    assert isinstance(info, dict)
                    assert "provider" in info
                    assert "dimension" in info
                    assert "total_vectors" in info
                    assert info["provider"] == "ollama"
                    assert info["dimension"] == 512

    def test_get_info_counts_vectors(self):
        """Test: get_info zählt Vektoren korrekt."""
        from src.embeddings.faiss_store import FaissStore

        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "count.index"
            map_file = Path(tmpdir) / "count.json"
            progress_file = Path(tmpdir) / "count_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                with patch.object(FaissStore, "__init__", lambda self, dim=None, provider="ollama": None):
                    store = FaissStore()
                    store.dim = None
                    store.index = None
                    store.id_map = {}
                    store.index_file = index_file
                    store.map_file = map_file
                    store.provider = "ollama"

                    # Füge 5 Vektoren hinzu
                    for i in range(5):
                        store.add([0.1 * i] * 128, doc_id=f"doc_{i}", persist_now=False)

                    info = store.get_info()
                    assert info["total_vectors"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
