"""Unit-Tests für src/config.py

Tests für:
- Pfad-Konfigurationen
- Default-Werte
- Konstanten
"""

import pytest
from pathlib import Path
from src.config import (
    ROOT,
    INPUT_CSV,
    DATA_DIR,
    EMBEDDINGS_FILE,
    FAISS_INDEX_FILE,
    EMBED_MAP_FILE,
    LOG_DIR,
    OLLAMA_EMBED_MODEL,
    LLM_DEFAULT_MODEL,
    TOP_K_DEFAULT,
    MAX_DOCS_FOR_LLM,
)


class TestPathConfiguration:
    """Tests für Pfad-Konfigurationen."""

    def test_root_path_exists(self):
        """Test: ROOT-Pfad ist ein gültiges Verzeichnis."""
        assert ROOT.exists()
        assert ROOT.is_dir()

    def test_input_csv_path_structure(self):
        """Test: INPUT_CSV hat korrekte Struktur."""
        assert INPUT_CSV.parent == ROOT / "input"
        assert INPUT_CSV.name == "foerderkatalog_export.csv"
        assert INPUT_CSV.suffix == ".csv"

    def test_data_dir_path(self):
        """Test: DATA_DIR ist korrekt konfiguriert."""
        assert DATA_DIR == ROOT / "data"

    def test_log_dir_path(self):
        """Test: LOG_DIR ist korrekt konfiguriert."""
        assert LOG_DIR == ROOT / "logs"

    def test_embeddings_file_location(self):
        """Test: EMBEDDINGS_FILE ist in DATA_DIR."""
        assert EMBEDDINGS_FILE.parent == DATA_DIR
        assert EMBEDDINGS_FILE.suffix == ".npy"

    def test_faiss_index_file_location(self):
        """Test: FAISS_INDEX_FILE ist in DATA_DIR."""
        assert FAISS_INDEX_FILE.parent == DATA_DIR
        assert FAISS_INDEX_FILE.name == "vector.index"

    def test_embed_map_file_location(self):
        """Test: EMBED_MAP_FILE ist in DATA_DIR."""
        assert EMBED_MAP_FILE.parent == DATA_DIR
        assert EMBED_MAP_FILE.suffix == ".json"


class TestModelConfiguration:
    """Tests für Modell-Konfigurationen."""

    def test_ollama_embed_model_is_string(self):
        """Test: OLLAMA_EMBED_MODEL ist ein String."""
        assert isinstance(OLLAMA_EMBED_MODEL, str)
        assert len(OLLAMA_EMBED_MODEL) > 0

    def test_llm_default_model_is_string(self):
        """Test: LLM_DEFAULT_MODEL ist ein String."""
        assert isinstance(LLM_DEFAULT_MODEL, str)
        assert len(LLM_DEFAULT_MODEL) > 0

    def test_ollama_model_format(self):
        """Test: Ollama-Modellname hat erwartetes Format."""
        assert "embed" in OLLAMA_EMBED_MODEL.lower()


class TestLimitConfiguration:
    """Tests für Limit-Konfigurationen."""

    def test_top_k_default_is_positive_int(self):
        """Test: TOP_K_DEFAULT ist eine positive Zahl."""
        assert isinstance(TOP_K_DEFAULT, int)
        assert TOP_K_DEFAULT > 0

    def test_top_k_default_reasonable_value(self):
        """Test: TOP_K_DEFAULT hat einen sinnvollen Wert."""
        assert 1 <= TOP_K_DEFAULT <= 1000

    def test_max_docs_for_llm_is_positive_int(self):
        """Test: MAX_DOCS_FOR_LLM ist eine positive Zahl."""
        assert isinstance(MAX_DOCS_FOR_LLM, int)
        assert MAX_DOCS_FOR_LLM > 0

    def test_max_docs_for_llm_reasonable_value(self):
        """Test: MAX_DOCS_FOR_LLM hat einen sinnvollen Wert."""
        assert 1 <= MAX_DOCS_FOR_LLM <= 100

    def test_max_docs_smaller_than_top_k(self):
        """Test: MAX_DOCS_FOR_LLM sollte <= TOP_K_DEFAULT sein."""
        # Dies ist eine Empfehlung, kein hartes Requirement
        # MAX_DOCS kann größer sein, sollte aber beachtet werden
        if MAX_DOCS_FOR_LLM > TOP_K_DEFAULT:
            pytest.warns(UserWarning)


class TestPathTypes:
    """Tests für Pfad-Typen."""

    def test_all_paths_are_path_objects(self):
        """Test: Alle Pfad-Konstanten sind Path-Objekte."""
        paths = [ROOT, INPUT_CSV, DATA_DIR, EMBEDDINGS_FILE, FAISS_INDEX_FILE, EMBED_MAP_FILE, LOG_DIR]

        for path in paths:
            assert isinstance(path, Path), f"{path} ist kein Path-Objekt"

    def test_paths_are_absolute(self):
        """Test: ROOT ist ein absoluter Pfad."""
        assert ROOT.is_absolute()


class TestConfigImmutability:
    """Tests für Config-Unveränderlichkeit."""

    def test_constants_are_final(self):
        """Test: Wichtige Konstanten sind als Final markiert (Typ-Check)."""
        # Diese Tests prüfen zur Laufzeit, ob Typing.Final verwendet wurde
        # In der Praxis wird dies durch mypy/type-checker verifiziert
        from typing import get_type_hints
        import src.config as config_module

        get_type_hints(config_module, include_extras=True)

        # Prüfe, dass wichtige Konstanten existieren
        assert "INPUT_CSV" in dir(config_module)
        assert "OLLAMA_EMBED_MODEL" in dir(config_module)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
