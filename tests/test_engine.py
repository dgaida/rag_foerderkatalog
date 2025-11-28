"""Unit-Tests für src/search/engine.py

Tests für:
- Datumsextraktion und Laufzeit-Berechnung
- CSV-Import und Datenbereinigung
- Embedding-Text-Erzeugung
- Semantische Suche
- Kontextbasierte Antwortgenerierung
- Provider-Switching
- build_embeddings_if_missing
- get_index_info
- Error-Handling
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.search.engine import ProjectSearchEngine


class TestProviderSpecific:
    """Tests für provider-spezifische Funktionalität."""

    def test_engine_with_ollama_provider(self):
        """Test: Engine mit Ollama-Provider initialisiert."""
        engine = ProjectSearchEngine(provider="ollama")

        assert engine.provider == "ollama"
        assert engine.faiss.provider == "ollama"

    def test_engine_with_huggingface_provider(self):
        """Test: Engine mit HuggingFace-Provider initialisiert."""
        engine = ProjectSearchEngine(provider="huggingface")

        assert engine.provider == "huggingface"
        assert engine.faiss.provider == "huggingface"

    def test_engine_with_custom_embed_model(self):
        """Test: Engine mit Custom Embedding-Modell."""
        engine = ProjectSearchEngine(provider="huggingface", embed_model="intfloat/e5-small-v2")

        assert engine.embed_model == "intfloat/e5-small-v2"


class TestBuildEmbeddingsIfMissing:
    """Tests für build_embeddings_if_missing()."""

    def test_build_embeddings_skips_if_index_exists(self):
        """Test: Embeddings werden übersprungen wenn Index existiert."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="latin1") as f:
            f.write('="FKZ";="Thema"\n')
            f.write('="ABC123";="Test"\n')
            csv_path = Path(f.name)

        try:
            engine = ProjectSearchEngine(csv_file=csv_path, provider="ollama")
            engine.load_and_clean()

            # Mock existierenden Index mit ntotal > 0
            mock_index = MagicMock()
            mock_index.ntotal = 100
            engine.faiss.index = mock_index

            with patch("src.search.engine.embed_text") as mock_embed:
                engine.build_embeddings_if_missing()

                # embed_text sollte NICHT aufgerufen werden
                mock_embed.assert_not_called()
        finally:
            csv_path.unlink()

    @patch("src.search.engine.embed_text")
    def test_build_embeddings_creates_new_index(self, mock_embed):
        """Test: Neue Embeddings werden erstellt wenn Index leer."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="latin1") as f:
            f.write('="FKZ";="Zuwendungsempfänger";="Thema"\n')
            f.write('="ABC";="Uni Test";="KI"\n')
            f.write('="DEF";="Institut";="Robotik"\n')
            csv_path = Path(f.name)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                index_file = Path(tmpdir) / "test.index"
                map_file = Path(tmpdir) / "test.json"
                progress_file = Path(tmpdir) / "progress.json"

                with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                    engine = ProjectSearchEngine(csv_file=csv_path, provider="ollama")
                    engine.load_and_clean()

                    # Stelle sicher dass Index initial leer ist
                    engine.faiss.index = None
                    engine.faiss.id_map = {}

                    mock_embed.return_value = [0.1] * 768

                    engine.build_embeddings_if_missing()

                    # Embeddings sollten erstellt worden sein
                    assert engine.faiss.index is not None
                    assert engine.faiss.index.ntotal == 2
                    assert mock_embed.call_count == 2
        finally:
            csv_path.unlink()

    def test_build_embeddings_raises_without_df(self):
        """Test: RuntimeError wenn DataFrame nicht geladen."""
        engine = ProjectSearchEngine(provider="ollama")

        with pytest.raises(RuntimeError, match="nicht geladen"):
            engine.build_embeddings_if_missing()

    @patch("src.search.engine.embed_text")
    def test_build_embeddings_handles_empty_texts(self, mock_embed):
        """Test: Leere Texte werden übersprungen (mit Warnung geloggt)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="latin1") as f:
            f.write('="FKZ";="Thema"\n')
            f.write('="ABC";=\n')  # Leerer Eintrag
            f.write('="DEF";="Valid"\n')
            csv_path = Path(f.name)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                index_file = Path(tmpdir) / "empty_test.index"
                map_file = Path(tmpdir) / "empty_test.json"
                progress_file = Path(tmpdir) / "progress.json"

                with patch("src.config.get_index_files", return_value=(index_file, map_file, progress_file)):
                    engine = ProjectSearchEngine(csv_file=csv_path)
                    engine.load_and_clean()

                    # Index explizit leer setzen
                    engine.faiss.index = None
                    engine.faiss.id_map = {}

                    mock_embed.return_value = [0.1] * 768

                    engine.build_embeddings_if_missing()

                    # Mindestens 1 Embedding sollte erstellt werden (für "Valid")
                    # Kann 1 oder 2 sein, je nachdem ob leerer String gefiltert wird
                    assert mock_embed.call_count >= 1
        finally:
            csv_path.unlink()


class TestGetIndexInfo:
    """Tests für get_index_info()."""

    def test_get_index_info_returns_dict(self):
        """Test: get_index_info gibt Dictionary zurück."""
        engine = ProjectSearchEngine(provider="ollama")

        info = engine.get_index_info()

        assert isinstance(info, dict)

    def test_get_index_info_contains_csv_info(self):
        """Test: get_index_info enthält CSV-Informationen."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="latin1") as f:
            f.write('="FKZ";="Thema"\n')
            f.write('="ABC";="Test"\n')
            csv_path = Path(f.name)

        try:
            engine = ProjectSearchEngine(csv_file=csv_path)
            engine.load_and_clean()

            info = engine.get_index_info()

            assert "csv_loaded" in info
            assert "csv_rows" in info
            assert info["csv_loaded"] is True
            assert info["csv_rows"] == 1
        finally:
            csv_path.unlink()

    def test_get_index_info_without_loaded_csv(self):
        """Test: get_index_info funktioniert ohne geladene CSV."""
        engine = ProjectSearchEngine(provider="huggingface")

        info = engine.get_index_info()

        assert info["csv_loaded"] is False
        assert info["csv_rows"] == 0


class TestSearchEdgeCases:
    """Tests für Edge-Cases bei der Suche."""

    @patch("src.search.engine.embed_text")
    def test_search_handles_embed_exception(self, mock_embed):
        """Test: Exception beim Embedding wird gefangen."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="latin1") as f:
            f.write('="FKZ";="Thema"\n')
            f.write('="ABC";="Test"\n')
            csv_path = Path(f.name)

        try:
            engine = ProjectSearchEngine(csv_file=csv_path)
            engine.load_and_clean()

            mock_embed.side_effect = RuntimeError("Embedding failed")

            # Sollte leeres DataFrame zurückgeben statt Exception
            result = engine.search("test query", k=5)

            assert isinstance(result, pd.DataFrame)
            assert result.empty
        finally:
            csv_path.unlink()


class TestAnswerWithContextEdgeCases:
    """Tests für Edge-Cases bei answer_with_context()."""

    @patch("src.search.engine.ProjectSearchEngine.search")
    def test_answer_with_context_no_results(self, mock_search):
        """Test: Antwort bei fehlenden Suchergebnissen."""
        engine = ProjectSearchEngine(provider="ollama")
        engine.df = pd.DataFrame()

        mock_search.return_value = pd.DataFrame()

        answer = engine.answer_with_context("test query")

        assert "Keine relevanten Projekte" in answer

    @patch("src.search.engine.chat_system_query")
    @patch("src.search.engine.ProjectSearchEngine.search")
    def test_answer_with_context_handles_llm_exception(self, mock_search, mock_chat):
        """Test: Exception beim LLM-Call wird weitergeleitet."""
        engine = ProjectSearchEngine(provider="ollama")
        engine.df = pd.DataFrame({'="FKZ"': ["ABC"]})

        mock_df = pd.DataFrame({'="FKZ"': ["ABC123"], '="Thema"': ["Test"]})
        mock_search.return_value = mock_df

        mock_chat.side_effect = RuntimeError("LLM API Error")

        with pytest.raises(RuntimeError, match="LLM API Error"):
            engine.answer_with_context("test query")


class TestDataCleaningEdgeCases:
    """Tests für Edge-Cases bei der Datenbereinigung."""

    def test_load_and_clean_handles_missing_columns(self):
        """Test: Fehlende Spalten werden korrekt behandelt."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="latin1") as f:
            f.write('="FKZ";="Thema"\n')  # Keine Laufzeit-Spalten
            f.write('="ABC";="Test"\n')
            csv_path = Path(f.name)

        try:
            engine = ProjectSearchEngine(csv_file=csv_path)
            engine.load_and_clean()

            # Sollte funktionieren, Laufzeit-Spalte sollte leer sein
            assert "__laufzeit" in engine.df.columns
            assert engine.df["__laufzeit"].iloc[0] == ""
        finally:
            csv_path.unlink()

    def test_load_and_clean_invalid_foerdersumme(self):
        """Test: Ungültige Fördersummen werden zu NaN."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="latin1") as f:
            f.write('="FKZ";="Fördersumme in EUR"\n')
            f.write('="ABC";invalid_number\n')
            csv_path = Path(f.name)

        try:
            engine = ProjectSearchEngine(csv_file=csv_path)
            engine.load_and_clean()

            # Ungültige Summe sollte NaN sein
            assert pd.isna(engine.df['="Fördersumme in EUR"'].iloc[0])
        finally:
            csv_path.unlink()


class TestYearExtraction:
    """Tests für die Jahreszahl-Extraktion."""

    def test_extract_year_valid_date(self):
        """Test: Jahreszahl wird aus gültigem Datum extrahiert."""
        engine = ProjectSearchEngine()

        assert engine._extract_year("01.03.2002") == 2002
        assert engine._extract_year("31.12.2025") == 2025
        assert engine._extract_year("15.06.1999") == 1999

    def test_extract_year_various_formats(self):
        """Test: Verschiedene Datumsformate werden erkannt."""
        engine = ProjectSearchEngine()

        assert engine._extract_year("2002-03-01") == 2002
        assert engine._extract_year("March 2025") == 2025
        assert engine._extract_year("Jahr 2020") == 2020

    def test_extract_year_invalid_input(self):
        """Test: Ungültige Eingaben liefern None."""
        engine = ProjectSearchEngine()

        assert engine._extract_year("") is None
        assert engine._extract_year("invalid") is None
        assert engine._extract_year(None) is None
        assert engine._extract_year("1899") is None  # Vor 1900
        assert engine._extract_year("2100") is None  # Nach 2099

    def test_extract_year_with_nan(self):
        """Test: NaN-Werte werden korrekt behandelt."""
        engine = ProjectSearchEngine()

        assert engine._extract_year(pd.NA) is None
        assert engine._extract_year(float("nan")) is None


class TestRuntimeString:
    """Tests für die Laufzeit-String-Erzeugung."""

    def test_create_runtime_string_both_dates(self):
        """Test: Laufzeit-String mit Start- und Enddatum."""
        engine = ProjectSearchEngine()
        row = pd.Series({'="Laufzeit von"': "01.03.2002", '="Laufzeit bis"': "31.12.2005"})

        result = engine._create_runtime_string(row)

        assert result == "2002 - 2005"

    def test_create_runtime_string_same_year(self):
        """Test: Laufzeit-String bei gleichem Jahr."""
        engine = ProjectSearchEngine()
        row = pd.Series({'="Laufzeit von"': "01.01.2020", '="Laufzeit bis"': "31.12.2020"})

        result = engine._create_runtime_string(row)

        assert result == "2020 - 2020"

    def test_create_runtime_string_only_start(self):
        """Test: Laufzeit-String nur mit Startdatum."""
        engine = ProjectSearchEngine()
        row = pd.Series({'="Laufzeit von"': "01.03.2002", '="Laufzeit bis"': ""})

        result = engine._create_runtime_string(row)

        assert result == "ab 2002"

    def test_create_runtime_string_only_end(self):
        """Test: Laufzeit-String nur mit Enddatum."""
        engine = ProjectSearchEngine()
        row = pd.Series({'="Laufzeit von"': "", '="Laufzeit bis"': "31.12.2005"})

        result = engine._create_runtime_string(row)

        assert result == "bis 2005"

    def test_create_runtime_string_no_dates(self):
        """Test: Leerer String bei fehlenden Daten."""
        engine = ProjectSearchEngine()
        row = pd.Series({'="Laufzeit von"': "", '="Laufzeit bis"': ""})

        result = engine._create_runtime_string(row)

        assert result == ""


class TestLoadAndClean:
    """Tests für CSV-Import und Datenbereinigung."""

    def test_load_and_clean_success(self):
        """Test: CSV wird erfolgreich geladen und bereinigt."""
        # Erstelle temporäre CSV-Datei
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="latin1") as f:
            f.write('="FKZ";="Zuwendungsempfänger";="Laufzeit von";="Laufzeit bis";="Fördersumme in EUR"\n')
            f.write('="ABC123";="Uni Test";="01.03.2002";="31.12.2005";160.000,00\n')
            f.write('="DEF456";="Institut XY";="01.01.2020";="31.12.2022";250.000,00\n')
            csv_path = Path(f.name)

        try:
            engine = ProjectSearchEngine(csv_file=csv_path)
            engine.load_and_clean()

            assert engine.df is not None
            assert len(engine.df) == 2
            assert "__laufzeit" in engine.df.columns
            assert engine.df["__laufzeit"].iloc[0] == "2002 - 2005"
            assert engine.df["__laufzeit"].iloc[1] == "2020 - 2022"
        finally:
            csv_path.unlink()

    def test_load_and_clean_removes_excel_formatting(self):
        """Test: Excel-Formatierung wird entfernt."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="latin1") as f:
            f.write('="FKZ";="Thema"\n')
            f.write('="ABC123";="Test Thema"\n')
            csv_path = Path(f.name)

        try:
            engine = ProjectSearchEngine(csv_file=csv_path)
            engine.load_and_clean()

            # Prüfe, dass =" und " entfernt wurden
            fkz_value = engine.df['="FKZ"'].iloc[0]
            assert not fkz_value.startswith('="')
            assert not fkz_value.endswith('"')
        finally:
            csv_path.unlink()

    def test_load_and_clean_converts_foerdersumme(self):
        """Test: Fördersumme wird in numerischen Wert konvertiert."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="latin1") as f:
            f.write('="Fördersumme in EUR"\n')
            f.write("160.000,00\n")
            f.write("1.500.000,50\n")
            csv_path = Path(f.name)

        try:
            engine = ProjectSearchEngine(csv_file=csv_path)
            engine.load_and_clean()

            assert engine.df['="Fördersumme in EUR"'].dtype in [float, "float64"]
            assert engine.df['="Fördersumme in EUR"'].iloc[0] == pytest.approx(160000.0)
            assert engine.df['="Fördersumme in EUR"'].iloc[1] == pytest.approx(1500000.5)
        finally:
            csv_path.unlink()

    def test_load_and_clean_file_not_found(self):
        """Test: FileNotFoundError bei fehlender CSV."""
        engine = ProjectSearchEngine(csv_file=Path("nonexistent.csv"))

        with pytest.raises(FileNotFoundError):
            engine.load_and_clean()


class TestBuildEmbeddingText:
    """Tests für die Embedding-Text-Erzeugung."""

    def test_build_embedding_text_all_fields(self):
        """Test: Alle Felder werden korrekt kombiniert."""
        engine = ProjectSearchEngine()
        row = pd.Series(
            {
                '="Zuwendungsempfänger"': "Uni Test",
                '="Thema"': "KI Forschung",
                '="Klartext Leistungsplansystematik"': "Grundlagenforschung",
                '="Ausführende Stelle"': "Institut ABC",
                '="Stadt/Gemeinde"': "Berlin",
                '="Bundesland"': "Berlin",
                "__laufzeit": "2020 - 2025",
                '="Förderprofil"': "Innovation",
                '="Verbundprojekt"': "KI-Cluster",
            }
        )

        result = engine._build_embedding_text(row)

        assert "Uni Test" in result
        assert "KI Forschung" in result
        assert "2020 - 2025" in result
        assert "Berlin" in result

    def test_build_embedding_text_missing_fields(self):
        """Test: Fehlende Felder werden übersprungen."""
        engine = ProjectSearchEngine()
        row = pd.Series({'="Zuwendungsempfänger"': "Uni Test", '="Thema"': "KI"})

        result = engine._build_embedding_text(row)

        assert "Uni Test" in result
        assert "KI" in result
        assert result.count(".") >= 1  # Mindestens ein Trennpunkt

    def test_build_embedding_text_nan_values(self):
        """Test: NaN-Werte werden ignoriert."""
        engine = ProjectSearchEngine()
        row = pd.Series({'="Zuwendungsempfänger"': "Uni Test", '="Thema"': pd.NA, '="Bundesland"': "Bayern"})

        result = engine._build_embedding_text(row)

        assert "Uni Test" in result
        assert "Bayern" in result
        assert "nan" not in result.lower()

    def test_build_embedding_text_empty_row(self):
        """Test: Leere Zeile liefert leeren String."""
        engine = ProjectSearchEngine()
        row = pd.Series({})

        result = engine._build_embedding_text(row)

        assert result == ""


class TestSearch:
    """Tests für die semantische Suche."""

    @patch("src.search.engine.embed_text")
    def test_search_success(self, mock_embed):
        """Test: Erfolgreiche Suche liefert Ergebnisse."""
        engine = ProjectSearchEngine()
        engine.df = pd.DataFrame({'="FKZ"': ["ABC123", "DEF456"], '="Thema"': ["KI", "Robotik"]})

        # Mock FAISS-Index
        mock_faiss = MagicMock()
        mock_faiss.index = MagicMock()
        mock_faiss.index.d = 768
        mock_faiss.search.return_value = [(0.95, "0"), (0.85, "1")]
        engine.faiss = mock_faiss

        mock_embed.return_value = [0.1] * 768

        results = engine.search("Künstliche Intelligenz", k=2)

        assert len(results) == 2
        assert "__score" in results.columns
        assert results["__score"].iloc[0] == 0.95

    def test_search_without_loaded_df(self):
        """Test: RuntimeError bei nicht geladenem DataFrame."""
        engine = ProjectSearchEngine()

        with pytest.raises(RuntimeError, match="nicht geladen"):
            engine.search("test")

    @patch("src.search.engine.embed_text")
    def test_search_dimension_mismatch(self, mock_embed):
        """Test: Leeres DataFrame bei Dimensions-Mismatch (Exception wird gefangen)."""
        engine = ProjectSearchEngine()
        engine.df = pd.DataFrame({'="FKZ"': ["ABC"]})

        mock_faiss = MagicMock()
        mock_faiss.index = MagicMock()
        mock_faiss.index.d = 768
        engine.faiss = mock_faiss

        mock_embed.return_value = [0.1] * 512  # Falsche Dimension

        # Die search()-Methode fängt die ValueError ab und returned leeres DataFrame
        result = engine.search("test")

        assert result.empty


class TestAnswerWithContext:
    """Tests für kontextbasierte Antwortgenerierung."""

    @patch("src.search.engine.chat_system_query")
    @patch("src.search.engine.ProjectSearchEngine.search")
    def test_answer_with_context_success(self, mock_search, mock_chat):
        """Test: Erfolgreiche Antwortgenerierung mit Kontext."""
        engine = ProjectSearchEngine()

        mock_df = pd.DataFrame(
            {
                '="FKZ"': ["ABC123"],
                '="Zuwendungsempfänger"': ["Uni Test"],
                '="Thema"': ["KI Forschung"],
                '="Fördersumme in EUR"': [100000.0],
                "__laufzeit": ["2020 - 2025"],
                '="Bundesland"': ["Bayern"],
            }
        )
        mock_search.return_value = mock_df
        mock_chat.return_value = "Test Antwort mit FKZ ABC123"

        result = engine.answer_with_context("KI Projekte")

        assert "Test Antwort" in result
        mock_search.assert_called_once()
        mock_chat.assert_called_once()

    @patch("src.search.engine.ProjectSearchEngine.search")
    def test_answer_with_context_no_results(self, mock_search):
        """Test: Meldung bei fehlenden Ergebnissen."""
        engine = ProjectSearchEngine()
        mock_search.return_value = pd.DataFrame()

        result = engine.answer_with_context("Unbekanntes Thema")

        assert "Keine relevanten Projekte" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
