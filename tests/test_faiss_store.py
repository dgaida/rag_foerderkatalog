"""Unit-Tests für src/embeddings/faiss_store.py

Tests für:
- FAISS-Index Initialisierung
- Hinzufügen von Vektoren
- Suche nach ähnlichen Vektoren
- Persistierung und Laden
- Provider-spezifische Funktionalität

FIXES:
1. Temporäre Verzeichnisse werden korrekt verwendet
2. Mock-Dateien werden nicht aus echten Verzeichnissen geladen
3. HuggingFace-Modell-Downloads werden gemockt
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
from src.embeddings.faiss_store import FaissStore


class TestFaissStoreInitialization:
    """Tests für FaissStore-Initialisierung."""

    def test_init_creates_store(self):
        """Test: FaissStore kann initialisiert werden."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "test_vector.index"
            map_file = Path(tmpdir) / "test_map.json"
            progress_file = Path(tmpdir) / "test_progress.json"

            # FIX: Patch get_index_files um tmp-Pfade zu verwenden
            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                store = FaissStore(dim=768, provider="ollama")

                assert store.dim == 768
                assert store.provider == "ollama"
                # FIX: id_map ist leer wenn keine Dateien existieren
                assert store.id_map == {}
                assert store.index is None  # Index existiert nicht

    def test_init_with_dimension(self):
        """Test: Initialisierung mit spezifischer Dimension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "test_hf.index"
            map_file = Path(tmpdir) / "test_hf.json"
            progress_file = Path(tmpdir) / "test_hf_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                store = FaissStore(dim=384, provider="huggingface")

                assert store.dim == 384
                assert store.provider == "huggingface"

    def test_init_loads_existing_index(self):
        """Test: Existierender Index wird geladen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Erstelle Mock-Index-Dateien
            index_file = Path(tmpdir) / "existing_vector.index"
            map_file = Path(tmpdir) / "existing_map.json"
            progress_file = Path(tmpdir) / "existing_progress.json"

            # Schreibe Test-Mapping
            test_map = {"0": "doc_0", "1": "doc_1"}
            map_file.write_text(json.dumps(test_map))

            # Erstelle FAISS-Index
            import faiss

            test_index = faiss.IndexFlatIP(768)
            faiss.write_index(test_index, str(index_file))

            # Patche get_index_files
            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                store = FaissStore(provider="ollama")

                assert store.index is not None
                assert store.dim == 768
                # FIX: Jetzt sollte id_map geladen werden
                assert store.id_map == test_map


class TestAddVector:
    """Tests für das Hinzufügen von Vektoren."""

    def test_add_single_vector(self):
        """Test: Einzelner Vektor wird hinzugefügt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "test_vector.index"
            map_file = Path(tmpdir) / "test_map.json"
            progress_file = Path(tmpdir) / "test_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                store = FaissStore(dim=768, provider="ollama")
                # Clear any existing index
                store.index = None
                store.id_map = {}

                vector = [0.1] * 768
                store.add(vector, doc_id="doc_0", persist_now=False)

                assert store.index is not None
                assert store.index.ntotal == 1
                assert "0" in store.id_map
                assert store.id_map["0"] == "doc_0"

    def test_add_multiple_vectors(self):
        """Test: Mehrere Vektoren werden hinzugefügt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "test_vector.index"
            map_file = Path(tmpdir) / "test_map.json"
            progress_file = Path(tmpdir) / "test_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                store = FaissStore(dim=384, provider="huggingface")
                store.index = None
                store.id_map = {}

                for i in range(5):
                    vector = [0.1 * (i + 1)] * 384
                    store.add(vector, doc_id=f"doc_{i}", persist_now=False)

                assert store.index.ntotal == 5
                assert len(store.id_map) == 5

    def test_add_empty_vector_raises_error(self):
        """Test: Leerer Vektor führt zu ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "test.index"
            map_file = Path(tmpdir) / "test.json"
            progress_file = Path(tmpdir) / "test_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                store = FaissStore(dim=768, provider="ollama")

                with pytest.raises(ValueError, match="Leerer Embedding-Vektor"):
                    store.add([], doc_id="doc_0")

    def test_add_vector_with_persist(self):
        """Test: Vektor mit sofortiger Persistierung."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "persist_vector.index"
            map_file = Path(tmpdir) / "persist_map.json"
            progress_file = Path(tmpdir) / "persist_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                store = FaissStore(dim=768, provider="ollama")
                store.index = None
                store.id_map = {}

                vector = [0.2] * 768
                store.add(vector, doc_id="test_doc", persist_now=True)

                # FIX: Prüfe dass Dateien im tmpdir erstellt wurden
                assert index_file.exists(), f"Index-Datei wurde nicht erstellt: {index_file}"
                assert map_file.exists(), f"Map-Datei wurde nicht erstellt: {map_file}"


class TestSearch:
    """Tests für die Vektorsuche."""

    def test_search_empty_index_returns_empty(self):
        """Test: Suche in leerem Index liefert leere Liste."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "empty.index"
            map_file = Path(tmpdir) / "empty.json"
            progress_file = Path(tmpdir) / "empty_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                store = FaissStore(dim=768, provider="ollama")
                store.index = None
                store.id_map = {}

                vector = [0.5] * 768
                results = store.search(vector, k=5)

                assert results == []

    def test_search_returns_correct_results(self):
        """Test: Suche liefert korrekte Ergebnisse."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "search.index"
            map_file = Path(tmpdir) / "search.json"
            progress_file = Path(tmpdir) / "search_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                store = FaissStore(dim=128, provider="ollama")
                store.index = None
                store.id_map = {}

                # Füge Test-Vektoren hinzu
                test_vectors = [
                    ([1.0] * 128, "doc_0"),
                    ([0.5] * 128, "doc_1"),
                    ([0.2] * 128, "doc_2"),
                ]

                for vec, doc_id in test_vectors:
                    store.add(vec, doc_id=doc_id, persist_now=False)

                # Suche nach ähnlichem Vektor
                query = [0.9] * 128
                results = store.search(query, k=2)

                assert len(results) == 2
                assert all(isinstance(r, tuple) for r in results)
                assert all(len(r) == 2 for r in results)
                # Erster Treffer sollte doc_0 sein (ähnlichster Vektor)
                assert results[0][1] == "doc_0"

    def test_search_respects_k_parameter(self):
        """Test: k-Parameter begrenzt Anzahl Ergebnisse."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "k_test.index"
            map_file = Path(tmpdir) / "k_test.json"
            progress_file = Path(tmpdir) / "k_test_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                store = FaissStore(dim=64, provider="ollama")
                store.index = None
                store.id_map = {}

                # Füge 10 Vektoren hinzu
                for i in range(10):
                    vector = [0.1 * i] * 64
                    store.add(vector, doc_id=f"doc_{i}", persist_now=False)

                # Suche mit k=3
                results = store.search([0.5] * 64, k=3)

                assert len(results) == 3


class TestPersistence:
    """Tests für Persistierung und Laden."""

    def test_persist_creates_files(self):
        """Test: Persist erstellt Index- und Map-Dateien."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "vector.index"
            map_file = Path(tmpdir) / "embeddings_map.json"
            progress_file = Path(tmpdir) / "progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                store = FaissStore(dim=768, provider="ollama")
                store.index = None
                store.id_map = {}

                # Füge Vektor hinzu
                store.add([0.3] * 768, doc_id="test", persist_now=False)

                # Persistiere
                store.persist()

                assert index_file.exists()
                assert map_file.exists()

    def test_persist_saves_correct_mapping(self):
        """Test: Mapping wird korrekt gespeichert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "vector.index"
            map_file = Path(tmpdir) / "embeddings_map.json"
            progress_file = Path(tmpdir) / "progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                store = FaissStore(dim=256, provider="huggingface")
                store.index = None
                store.id_map = {}

                store.add([0.1] * 256, doc_id="doc_a", persist_now=False)
                store.add([0.2] * 256, doc_id="doc_b", persist_now=False)

                store.persist()

                # Lade Mapping
                saved_map = json.loads(map_file.read_text())

                assert "0" in saved_map
                assert "1" in saved_map
                assert saved_map["0"] == "doc_a"
                assert saved_map["1"] == "doc_b"


class TestClear:
    """Tests für das Löschen des Index."""

    def test_clear_removes_index(self):
        """Test: Clear entfernt Index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "clear.index"
            map_file = Path(tmpdir) / "clear.json"
            progress_file = Path(tmpdir) / "clear_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                store = FaissStore(dim=128, provider="ollama")
                store.index = None
                store.id_map = {}

                store.add([0.5] * 128, doc_id="doc", persist_now=False)
                assert store.index is not None

                store.clear()

                assert store.index is None
                assert store.id_map == {}


class TestGetInfo:
    """Tests für get_info()."""

    def test_get_info_returns_dict(self):
        """Test: get_info gibt Dictionary zurück."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "info.index"
            map_file = Path(tmpdir) / "info.json"
            progress_file = Path(tmpdir) / "info_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                store = FaissStore(dim=768, provider="ollama")

                info = store.get_info()

                assert isinstance(info, dict)

    def test_get_info_contains_required_keys(self):
        """Test: get_info enthält alle erforderlichen Keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "keys.index"
            map_file = Path(tmpdir) / "keys.json"
            progress_file = Path(tmpdir) / "keys_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                store = FaissStore(dim=384, provider="huggingface")

                info = store.get_info()

                required_keys = [
                    "provider",
                    "index_file",
                    "map_file",
                    "dimension",
                    "total_vectors",
                    "exists",
                ]
                for key in required_keys:
                    assert key in info

    def test_get_info_correct_values(self):
        """Test: get_info liefert korrekte Werte."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "values.index"
            map_file = Path(tmpdir) / "values.json"
            progress_file = Path(tmpdir) / "values_progress.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                store = FaissStore(dim=512, provider="ollama")
                store.index = None
                store.id_map = {}

                store.add([0.1] * 512, doc_id="test1", persist_now=False)
                store.add([0.2] * 512, doc_id="test2", persist_now=False)

                info = store.get_info()

                assert info["provider"] == "ollama"
                assert info["dimension"] == 512
                assert info["total_vectors"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
