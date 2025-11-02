"""Unit-Tests für src/search/engine.py

Tests für:
- Datumsextraktion und Laufzeit-Berechnung
- CSV-Import und Datenbereinigung
- Embedding-Text-Erzeugung
- Semantische Suche
- Kontextbasierte Antwortgenerierung
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.search.engine import ProjectSearchEngine


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
