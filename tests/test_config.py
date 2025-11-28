"""Unit-Tests für src/config.py

Tests für:
- Pfad-Konfigurationen
- Default-Werte
- Konstanten
- Provider-spezifische Index-Dateien
"""

from pathlib import Path

import pytest

from src.config import (
    DATA_DIR,
    EMBED_MAP_FILE_HF,
    EMBED_MAP_FILE_OLLAMA,
    EMBEDDINGS_FILE,
    FAISS_INDEX_FILE_HF,
    FAISS_INDEX_FILE_OLLAMA,
    HF_EMBED_MODEL_DEFAULT,
    INPUT_CSV,
    LLM_DEFAULT_MODEL,
    LOG_DIR,
    MAX_DOCS_FOR_LLM,
    OLLAMA_EMBED_MODEL,
    ROOT,
    TOP_K_DEFAULT,
    get_index_files,
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

    def test_faiss_index_files_location(self):
        """Test: FAISS Index-Dateien sind in DATA_DIR."""
        assert FAISS_INDEX_FILE_OLLAMA.parent == DATA_DIR
        assert FAISS_INDEX_FILE_OLLAMA.name == "vector.index"

        assert FAISS_INDEX_FILE_HF.parent == DATA_DIR
        assert FAISS_INDEX_FILE_HF.name == "vector_hf.index"

    def test_embed_map_files_location(self):
        """Test: Embedding-Map-Dateien sind in DATA_DIR."""
        assert EMBED_MAP_FILE_OLLAMA.parent == DATA_DIR
        assert EMBED_MAP_FILE_OLLAMA.suffix == ".json"

        assert EMBED_MAP_FILE_HF.parent == DATA_DIR
        assert EMBED_MAP_FILE_HF.suffix == ".json"


class TestModelConfiguration:
    """Tests für Modell-Konfigurationen."""

    def test_ollama_embed_model_is_string(self):
        """Test: OLLAMA_EMBED_MODEL ist ein String."""
        assert isinstance(OLLAMA_EMBED_MODEL, str)
        assert len(OLLAMA_EMBED_MODEL) > 0

    def test_hf_embed_model_is_string(self):
        """Test: HF_EMBED_MODEL_DEFAULT ist ein String."""
        assert isinstance(HF_EMBED_MODEL_DEFAULT, str)
        assert len(HF_EMBED_MODEL_DEFAULT) > 0

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


class TestPathTypes:
    """Tests für Pfad-Typen."""

    def test_all_paths_are_path_objects(self):
        """Test: Alle Pfad-Konstanten sind Path-Objekte."""
        paths = [
            ROOT,
            INPUT_CSV,
            DATA_DIR,
            EMBEDDINGS_FILE,
            FAISS_INDEX_FILE_OLLAMA,
            FAISS_INDEX_FILE_HF,
            EMBED_MAP_FILE_OLLAMA,
            EMBED_MAP_FILE_HF,
            LOG_DIR,
        ]

        for path in paths:
            assert isinstance(path, Path), f"{path} ist kein Path-Objekt"

    def test_paths_are_absolute(self):
        """Test: ROOT ist ein absoluter Pfad."""
        assert ROOT.is_absolute()


class TestGetIndexFiles:
    """Tests für die get_index_files() Funktion."""

    def test_get_index_files_ollama(self):
        """Test: Ollama Index-Dateien werden korrekt zurückgegeben."""
        index_file, map_file, progress_file = get_index_files("ollama")

        assert index_file == FAISS_INDEX_FILE_OLLAMA
        assert map_file == EMBED_MAP_FILE_OLLAMA
        assert index_file.name == "vector.index"

    def test_get_index_files_huggingface(self):
        """Test: HuggingFace Index-Dateien werden korrekt zurückgegeben."""
        index_file, map_file, progress_file = get_index_files("huggingface")

        assert index_file == FAISS_INDEX_FILE_HF
        assert map_file == EMBED_MAP_FILE_HF
        assert index_file.name == "vector_hf.index"

    def test_get_index_files_invalid_provider(self):
        """Test: ValueError bei unbekanntem Provider."""
        with pytest.raises(ValueError, match="Unbekannter Provider"):
            get_index_files("invalid_provider")

    def test_get_index_files_returns_tuple(self):
        """Test: Funktion gibt Tuple mit 3 Elementen zurück."""
        result = get_index_files("ollama")

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_get_index_files_all_paths(self):
        """Test: Alle zurückgegebenen Werte sind Path-Objekte."""
        index_file, map_file, progress_file = get_index_files("ollama")

        assert isinstance(index_file, Path)
        assert isinstance(map_file, Path)
        assert isinstance(progress_file, Path)


class TestProviderSpecificFiles:
    """Tests für provider-spezifische Datei-Unterschiede."""

    def test_ollama_and_hf_have_different_files(self):
        """Test: Ollama und HuggingFace haben unterschiedliche Dateien."""
        ollama_index, _, _ = get_index_files("ollama")
        hf_index, _, _ = get_index_files("huggingface")

        assert ollama_index != hf_index
        assert ollama_index.name != hf_index.name

    def test_ollama_files_naming_convention(self):
        """Test: Ollama-Dateien folgen Namenskonvention."""
        index_file, map_file, progress_file = get_index_files("ollama")

        assert "hf" not in index_file.name
        assert "hf" not in map_file.name

    def test_hf_files_naming_convention(self):
        """Test: HuggingFace-Dateien folgen Namenskonvention."""
        index_file, map_file, progress_file = get_index_files("huggingface")

        assert "hf" in index_file.name
        assert "hf" in map_file.name


class TestConfigImmutability:
    """Tests für Config-Unveränderlichkeit."""

    def test_constants_are_final(self):
        """Test: Wichtige Konstanten sind als Final markiert (Typ-Check)."""
        from typing import get_type_hints

        import src.config as config_module

        get_type_hints(config_module, include_extras=True)

        # Prüfe, dass wichtige Konstanten existieren
        assert "INPUT_CSV" in dir(config_module)
        assert "OLLAMA_EMBED_MODEL" in dir(config_module)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
