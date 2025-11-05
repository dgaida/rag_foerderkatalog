"""Unit-Tests für src/embeddings/faiss_store.py

Tests für:
- FAISS-Index Initialisierung
- Hinzufügen von Vektoren
- Suche nach ähnlichen Vektoren
- Persistierung und Laden
- Provider-spezifische Funktionalität
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
from src.embeddings.faiss_store import FaissStore


class TestFaissStoreInitialization:
    """Tests für FaissStore-Initialisierung."""

    def test_init_without_existing_index(self):
        """Test: Initialisierung ohne existierenden Index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.embeddings.faiss_store.DATA_DIR", Path(tmpdir)):
                store = FaissStore(dim=768, provider="ollama")

                assert store.dim == 768
                assert store.provider == "ollama"
                assert store.index is None
                assert store.id_map == {}

    def test_init_with_dimension(self):
        """Test: Initialisierung mit spezifischer Dimension."""
        store = FaissStore(dim=384, provider="huggingface")

        assert store.dim == 384
        assert store.provider == "huggingface"

    def test_init_loads_existing_index(self):
        """Test: Existierender Index wird geladen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Erstelle Mock-Index-Dateien
            index_file = Path(tmpdir) / "vector.index"
            map_file = Path(tmpdir) / "embeddings_map.json"

            # Schreibe Test-Mapping
            test_map = {"0": "doc_0", "1": "doc_1"}
            map_file.write_text(json.dumps(test_map))

            # Erstelle FAISS-Index
            import faiss

            test_index = faiss.IndexFlatIP(768)
            faiss.write_index(test_index, str(index_file))

            # Patche Pfade
            with patch("src.embeddings.faiss_store.DATA_DIR", Path(tmpdir)):
                with patch("src.config.FAISS_INDEX_FILE_OLLAMA", index_file):
                    with patch("src.config.EMBED_MAP_FILE_OLLAMA", map_file):
                        with patch(
                            "src.config.get_index_files", return_value=(index_file, map_file, Path(tmpdir) / "progress.json")
                        ):
                            store = FaissStore(provider="ollama")

                            assert store.index is not None
                            assert store.dim == 768
                            assert store.id_map == test_map


class TestAddVector:
    """Tests für das Hinzufügen von Vektoren."""

    def test_add_single_vector(self):
        """Test: Einzelner Vektor wird hinzugefügt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.embeddings.faiss_store.DATA_DIR", Path(tmpdir)):
                store = FaissStore(dim=768, provider="ollama")

                vector = [0.1] * 768
                store.add(vector, doc_id="doc_0", persist_now=False)

                assert store.index is not None
                assert store.index.ntotal == 1
                assert "0" in store.id_map
                assert store.id_map["0"] == "doc_0"

    def test_add_multiple_vectors(self):
        """Test: Mehrere Vektoren werden hinzugefügt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.embeddings.faiss_store.DATA_DIR", Path(tmpdir)):
                store = FaissStore(dim=384, provider="huggingface")

                for i in range(5):
                    vector = [0.1 * (i + 1)] * 384
                    store.add(vector, doc_id=f"doc_{i}", persist_now=False)

                assert store.index.ntotal == 5
                assert len(store.id_map) == 5

    def test_add_empty_vector_raises_error(self):
        """Test: Leerer Vektor führt zu ValueError."""
        store = FaissStore(dim=768, provider="ollama")

        with pytest.raises(ValueError, match="Leerer Embedding-Vektor"):
            store.add([], doc_id="doc_0")

    def test_add_vector_with_persist(self):
        """Test: Vektor mit sofortiger Persistierung."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.embeddings.faiss_store.DATA_DIR", Path(tmpdir)):
                index_file = Path(tmpdir) / "vector.index"
                map_file = Path(tmpdir) / "embeddings_map.json"

                with patch("src.config.get_index_files", return_value=(index_file, map_file, Path(tmpdir) / "progress.json")):
                    store = FaissStore(dim=768, provider="ollama")
                    store.index_file = index_file
                    store.map_file = map_file

                    vector = [0.2] * 768
                    store.add(vector, doc_id="test_doc", persist_now=True)

                    # Prüfe dass Dateien erstellt wurden
                    assert index_file.exists()
                    assert map_file.exists()


class TestSearch:
    """Tests für die Vektorsuche."""

    def test_search_empty_index_returns_empty(self):
        """Test: Suche in leerem Index liefert leere Liste."""
        store = FaissStore(dim=768, provider="ollama")

        vector = [0.5] * 768
        results = store.search(vector, k=5)

        assert results == []

    def test_search_returns_correct_results(self):
        """Test: Suche liefert korrekte Ergebnisse."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.embeddings.faiss_store.DATA_DIR", Path(tmpdir)):
                store = FaissStore(dim=128, provider="ollama")

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
            with patch("src.embeddings.faiss_store.DATA_DIR", Path(tmpdir)):
                store = FaissStore(dim=64, provider="ollama")

                # Füge 10 Vektoren hinzu
                for i in range(10):
                    vector = [0.1 * i] * 64
                    store.add(vector, doc_id=f"doc_{i}", persist_now=False)

                # Suche mit k=3
                results = store.search([0.5] * 64, k=3)

                assert len(results) == 3

    def test_search_handles_k_larger_than_index(self):
        """Test: k größer als Index-Größe wird angepasst."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.embeddings.faiss_store.DATA_DIR", Path(tmpdir)):
                store = FaissStore(dim=64, provider="ollama")

                # Nur 2 Vektoren
                store.add([0.1] * 64, doc_id="doc_0", persist_now=False)
                store.add([0.2] * 64, doc_id="doc_1", persist_now=False)

                # Suche mit k=10 (größer als verfügbar)
                results = store.search([0.15] * 64, k=10)

                # Sollte nur 2 Ergebnisse liefern
                assert len(results) == 2


class TestPersistence:
    """Tests für Persistierung und Laden."""

    def test_persist_creates_files(self):
        """Test: Persist erstellt Index- und Map-Dateien."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "vector.index"
            map_file = Path(tmpdir) / "embeddings_map.json"

            with patch("src.config.get_index_files", return_value=(index_file, map_file, Path(tmpdir) / "progress.json")):
                store = FaissStore(dim=768, provider="ollama")
                store.index_file = index_file
                store.map_file = map_file

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

            with patch("src.config.get_index_files", return_value=(index_file, map_file, Path(tmpdir) / "progress.json")):
                store = FaissStore(dim=256, provider="huggingface")
                store.index_file = index_file
                store.map_file = map_file

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
            with patch("src.embeddings.faiss_store.DATA_DIR", Path(tmpdir)):
                store = FaissStore(dim=128, provider="ollama")

                store.add([0.5] * 128, doc_id="doc", persist_now=False)
                assert store.index is not None

                store.clear()

                assert store.index is None
                assert store.id_map == {}

    def test_clear_removes_files(self):
        """Test: Clear löscht persistierte Dateien."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "vector.index"
            map_file = Path(tmpdir) / "embeddings_map.json"

            # Erstelle Dateien
            index_file.touch()
            map_file.touch()

            with patch("src.config.get_index_files", return_value=(index_file, map_file, Path(tmpdir) / "progress.json")):
                store = FaissStore(provider="ollama")
                store.index_file = index_file
                store.map_file = map_file

                store.clear()

                assert not index_file.exists()
                assert not map_file.exists()


class TestGetInfo:
    """Tests für get_info()."""

    def test_get_info_returns_dict(self):
        """Test: get_info gibt Dictionary zurück."""
        store = FaissStore(dim=768, provider="ollama")

        info = store.get_info()

        assert isinstance(info, dict)

    def test_get_info_contains_required_keys(self):
        """Test: get_info enthält alle erforderlichen Keys."""
        store = FaissStore(dim=384, provider="huggingface")

        info = store.get_info()

        required_keys = ["provider", "index_file", "map_file", "dimension", "total_vectors", "exists"]
        for key in required_keys:
            assert key in info

    def test_get_info_correct_values(self):
        """Test: get_info liefert korrekte Werte."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.embeddings.faiss_store.DATA_DIR", Path(tmpdir)):
                store = FaissStore(dim=512, provider="ollama")

                store.add([0.1] * 512, doc_id="test1", persist_now=False)
                store.add([0.2] * 512, doc_id="test2", persist_now=False)

                info = store.get_info()

                assert info["provider"] == "ollama"
                assert info["dimension"] == 512
                assert info["total_vectors"] == 2


class TestProviderSpecific:
    """Tests für provider-spezifische Funktionalität."""

    def test_ollama_provider_uses_correct_files(self):
        """Test: Ollama-Provider verwendet korrekte Dateien."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ollama_index = Path(tmpdir) / "vector.index"
            ollama_map = Path(tmpdir) / "embeddings_map.json"
            ollama_progress = Path(tmpdir) / "indexing_progress.json"

            with patch("src.config.get_index_files", return_value=(ollama_index, ollama_map, ollama_progress)):
                store = FaissStore(provider="ollama")

                assert store.index_file == ollama_index
                assert store.map_file == ollama_map

    def test_huggingface_provider_uses_correct_files(self):
        """Test: HuggingFace-Provider verwendet korrekte Dateien."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hf_index = Path(tmpdir) / "vector_hf.index"
            hf_map = Path(tmpdir) / "embeddings_map_hf.json"
            hf_progress = Path(tmpdir) / "indexing_progress_hf.json"

            with patch("src.config.get_index_files", return_value=(hf_index, hf_map, hf_progress)):
                store = FaissStore(provider="huggingface")

                assert store.index_file == hf_index
                assert store.map_file == hf_map


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
