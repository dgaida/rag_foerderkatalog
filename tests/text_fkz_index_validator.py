"""Unit-Tests für src/utils/fkz_index_validator.py

Tests für:
- FKZ-basierte Index-Validierung
- Identifikation neuer Projekte
- Identifikation entfernter Projekte
- CSV-Vergleich
- Statistik-Generierung
- Error-Handling
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.embeddings.faiss_store import FaissStore
from src.utils.fkz_index_validator import FKZIndexValidator, compare_csv_files, get_projects_to_index, validate_with_fkz

# ===== Fixtures =====


@pytest.fixture
def sample_df():
    """Erstellt einen Test-DataFrame mit FKZ-Spalte."""
    return pd.DataFrame(
        {
            '="FKZ"': ["ABC001", "ABC002", "ABC003", "ABC004", "ABC005"],
            '="Zuwendungsempfänger"': ["Uni A", "Uni B", "Uni C", "Uni D", "Uni E"],
            '="Thema"': ["KI", "Robotik", "Energie", "Medizin", "Klima"],
        }
    )


@pytest.fixture
def mock_faiss_empty():
    """Erstellt einen leeren Mock-FAISS-Store."""
    mock = MagicMock(spec=FaissStore)
    mock.index = None
    mock.id_map = {}
    return mock


@pytest.fixture
def mock_faiss_with_data(sample_df):
    """Erstellt einen Mock-FAISS-Store mit einigen FKZ."""
    mock = MagicMock(spec=FaissStore)
    mock.index = MagicMock()
    mock.index.ntotal = 3
    # Index enthält die ersten 3 Zeilen (0, 1, 2) -> FKZ: ABC001, ABC002, ABC003
    mock.id_map = {"0": "0", "1": "1", "2": "2"}
    return mock


# ===== Test-Klassen =====


class TestFKZIndexValidatorInitialization:
    """Tests für die Initialisierung des FKZIndexValidator."""

    def test_init_with_valid_data(self, mock_faiss_empty, sample_df):
        """Test: Validator wird korrekt initialisiert."""
        validator = FKZIndexValidator(mock_faiss_empty, sample_df)

        assert validator.faiss == mock_faiss_empty
        assert validator.df.equals(sample_df)
        assert validator.fkz_column == '="FKZ"'

    def test_init_with_custom_fkz_column(self, mock_faiss_empty):
        """Test: Custom FKZ-Spaltenname wird akzeptiert."""
        df = pd.DataFrame({"CustomFKZ": ["A", "B"], "Data": [1, 2]})

        validator = FKZIndexValidator(mock_faiss_empty, df, fkz_column="CustomFKZ")

        assert validator.fkz_column == "CustomFKZ"

    def test_init_raises_on_missing_fkz_column(self, mock_faiss_empty):
        """Test: ValueError wenn FKZ-Spalte fehlt."""
        df = pd.DataFrame({"Data": [1, 2, 3]})

        with pytest.raises(ValueError, match="FKZ-Spalte.*nicht in DataFrame gefunden"):
            FKZIndexValidator(mock_faiss_empty, df, fkz_column='="FKZ"')


class TestGetIndexedFKZ:
    """Tests für get_indexed_fkz()."""

    def test_get_indexed_fkz_empty_index(self, mock_faiss_empty, sample_df):
        """Test: Leerer Index gibt leere Menge zurück."""
        validator = FKZIndexValidator(mock_faiss_empty, sample_df)

        result = validator.get_indexed_fkz()

        assert result == set()

    def test_get_indexed_fkz_with_data(self, mock_faiss_with_data, sample_df):
        """Test: Indizierte FKZ werden korrekt extrahiert."""
        validator = FKZIndexValidator(mock_faiss_with_data, sample_df)

        result = validator.get_indexed_fkz()

        assert result == {"ABC001", "ABC002", "ABC003"}

    def test_get_indexed_fkz_handles_invalid_indices(self, sample_df):
        """Test: Ungültige Indizes werden übersprungen."""
        mock_faiss = MagicMock(spec=FaissStore)
        mock_faiss.id_map = {
            "0": "0",  # Valid
            "1": "999",  # Out of bounds
            "2": "invalid",  # Not an int
        }

        validator = FKZIndexValidator(mock_faiss, sample_df)
        result = validator.get_indexed_fkz()

        # Nur der gültige Index 0 sollte ein FKZ liefern
        assert "ABC001" in result
        assert len(result) == 1

    def test_get_indexed_fkz_filters_nan_values(self, mock_faiss_empty):
        """Test: NaN-FKZ-Werte werden gefiltert."""
        df = pd.DataFrame({'="FKZ"': ["ABC001", float("nan"), "ABC003"]})

        mock_faiss = MagicMock(spec=FaissStore)
        mock_faiss.id_map = {"0": "0", "1": "1", "2": "2"}

        validator = FKZIndexValidator(mock_faiss, df)
        result = validator.get_indexed_fkz()

        assert result == {"ABC001", "ABC003"}
        assert len(result) == 2


class TestGetCSVFKZ:
    """Tests für get_csv_fkz()."""

    def test_get_csv_fkz_normal_dataframe(self, mock_faiss_empty, sample_df):
        """Test: FKZ aus CSV werden korrekt extrahiert."""
        validator = FKZIndexValidator(mock_faiss_empty, sample_df)

        result = validator.get_csv_fkz()

        assert result == {"ABC001", "ABC002", "ABC003", "ABC004", "ABC005"}

    def test_get_csv_fkz_empty_dataframe(self, mock_faiss_empty):
        """Test: Leerer DataFrame gibt leere Menge zurück."""
        df = pd.DataFrame({'="FKZ"': []})

        validator = FKZIndexValidator(mock_faiss_empty, df)
        result = validator.get_csv_fkz()

        assert result == set()

    def test_get_csv_fkz_strips_whitespace(self, mock_faiss_empty):
        """Test: Whitespace wird entfernt."""
        df = pd.DataFrame({'="FKZ"': ["  ABC001  ", "ABC002\n", "\tABC003"]})

        validator = FKZIndexValidator(mock_faiss_empty, df)
        result = validator.get_csv_fkz()

        assert result == {"ABC001", "ABC002", "ABC003"}

    def test_get_csv_fkz_filters_empty_strings(self, mock_faiss_empty):
        """Test: Leere Strings werden gefiltert."""
        df = pd.DataFrame({'="FKZ"': ["ABC001", "", "ABC003", "   "]})

        validator = FKZIndexValidator(mock_faiss_empty, df)
        result = validator.get_csv_fkz()

        assert result == {"ABC001", "ABC003"}

    def test_get_csv_fkz_filters_nan(self, mock_faiss_empty):
        """Test: NaN-Werte werden gefiltert."""
        df = pd.DataFrame({'="FKZ"': ["ABC001", pd.NA, "ABC003", float("nan")]})

        validator = FKZIndexValidator(mock_faiss_empty, df)
        result = validator.get_csv_fkz()

        assert result == {"ABC001", "ABC003"}


class TestGetNewFKZ:
    """Tests für get_new_fkz()."""

    def test_get_new_fkz_all_new(self, mock_faiss_empty, sample_df):
        """Test: Alle FKZ sind neu wenn Index leer."""
        validator = FKZIndexValidator(mock_faiss_empty, sample_df)

        result = validator.get_new_fkz()

        assert result == {"ABC001", "ABC002", "ABC003", "ABC004", "ABC005"}

    def test_get_new_fkz_some_new(self, mock_faiss_with_data, sample_df):
        """Test: Nur nicht-indizierte FKZ werden identifiziert."""
        validator = FKZIndexValidator(mock_faiss_with_data, sample_df)

        result = validator.get_new_fkz()

        assert result == {"ABC004", "ABC005"}

    def test_get_new_fkz_none_new(self, sample_df):
        """Test: Keine neuen FKZ wenn alle indiziert."""
        mock_faiss = MagicMock(spec=FaissStore)
        mock_faiss.id_map = {str(i): str(i) for i in range(5)}

        validator = FKZIndexValidator(mock_faiss, sample_df)
        result = validator.get_new_fkz()

        assert result == set()


class TestGetRemovedFKZ:
    """Tests für get_removed_fkz()."""

    def test_get_removed_fkz_none_removed(self, mock_faiss_with_data, sample_df):
        """Test: Keine entfernten FKZ."""
        validator = FKZIndexValidator(mock_faiss_with_data, sample_df)

        result = validator.get_removed_fkz()

        assert result == set()

    def test_get_removed_fkz_some_removed(self, sample_df):
        """Test: Entfernte FKZ werden identifiziert."""
        # Index enthält FKZ die nicht mehr in CSV sind
        mock_faiss = MagicMock(spec=FaissStore)

        # Erweitere DataFrame um Index 5 und 6 für die "alten" FKZ
        extended_df = pd.concat(
            [sample_df, pd.DataFrame({'="FKZ"': ["OLD001", "OLD002"], '="Zuwendungsempfänger"': ["Old A", "Old B"]})],
            ignore_index=True,
        )

        # id_map verweist auf Indizes 5 und 6 (OLD001, OLD002)
        mock_faiss.id_map = {"0": "5", "1": "6"}

        validator = FKZIndexValidator(mock_faiss, extended_df)

        # Jetzt simulieren wir, dass CSV nur sample_df ist (ohne OLD001, OLD002)
        validator.df = sample_df

        result = validator.get_removed_fkz()

        assert result == {"OLD001", "OLD002"}


class TestGetNewProjects:
    """Tests für get_new_projects()."""

    def test_get_new_projects_returns_dataframe(self, mock_faiss_with_data, sample_df):
        """Test: DataFrame mit neuen Projekten wird zurückgegeben."""
        validator = FKZIndexValidator(mock_faiss_with_data, sample_df)

        result = validator.get_new_projects()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "ABC004" in result['="FKZ"'].values
        assert "ABC005" in result['="FKZ"'].values

    def test_get_new_projects_with_limit(self, mock_faiss_with_data, sample_df):
        """Test: Limit begrenzt Anzahl der Projekte."""
        validator = FKZIndexValidator(mock_faiss_with_data, sample_df)

        result = validator.get_new_projects(limit=1)

        assert len(result) == 1

    def test_get_new_projects_no_new(self, sample_df):
        """Test: Leeres DataFrame wenn keine neuen Projekte."""
        mock_faiss = MagicMock(spec=FaissStore)
        mock_faiss.id_map = {str(i): str(i) for i in range(5)}

        validator = FKZIndexValidator(mock_faiss, sample_df)
        result = validator.get_new_projects()

        assert result.empty


class TestValidate:
    """Tests für validate()."""

    def test_validate_empty_index(self, mock_faiss_empty, sample_df):
        """Test: Validierung mit leerem Index."""
        validator = FKZIndexValidator(mock_faiss_empty, sample_df)

        is_complete, stats = validator.validate()

        assert is_complete is False
        assert stats["csv_total"] == 5
        assert stats["indexed_total"] == 0
        assert stats["new_count"] == 5
        assert stats["removed_count"] == 0
        assert stats["is_empty"] is True
        assert stats["sync_percentage"] == 0.0

    def test_validate_partial_index(self, mock_faiss_with_data, sample_df):
        """Test: Validierung mit teilweise gefülltem Index."""
        validator = FKZIndexValidator(mock_faiss_with_data, sample_df)

        is_complete, stats = validator.validate()

        assert is_complete is False
        assert stats["csv_total"] == 5
        assert stats["indexed_total"] == 3
        assert stats["new_count"] == 2
        assert stats["sync_percentage"] == 60.0  # 3/5 = 60%

    def test_validate_complete_index(self, sample_df):
        """Test: Validierung mit vollständigem Index."""
        mock_faiss = MagicMock(spec=FaissStore)
        mock_faiss.id_map = {str(i): str(i) for i in range(5)}

        validator = FKZIndexValidator(mock_faiss, sample_df)
        is_complete, stats = validator.validate()

        assert is_complete is True
        assert stats["new_count"] == 0
        assert stats["sync_percentage"] == 100.0

    def test_validate_returns_limited_fkz_lists(self, mock_faiss_empty):
        """Test: FKZ-Listen in Stats sind auf 100 begrenzt."""
        # Erstelle DataFrame mit 150 FKZ
        df = pd.DataFrame({'="FKZ"': [f"FKZ{i:05d}" for i in range(150)]})

        validator = FKZIndexValidator(mock_faiss_empty, df)
        is_complete, stats = validator.validate()

        assert len(stats["new_fkz"]) <= 100
        assert stats["new_count"] == 150


class TestCompareCSVFiles:
    """Tests für compare_csv_files()."""

    def test_compare_csv_files_no_changes(self):
        """Test: Vergleich identischer CSV-Dateien."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv1 = Path(tmpdir) / "file1.csv"
            csv2 = Path(tmpdir) / "file2.csv"

            # Erstelle identische CSVs
            df = pd.DataFrame({'="FKZ"': ["ABC001", "ABC002"], "Data": [1, 2]})
            df.to_csv(csv1, sep=";", index=False, encoding="latin1")
            df.to_csv(csv2, sep=";", index=False, encoding="latin1")

            stats = compare_csv_files(csv1, csv2)

            assert stats["added_count"] == 0
            assert stats["removed_count"] == 0
            assert stats["unchanged_count"] == 2

    def test_compare_csv_files_with_additions(self):
        """Test: CSV mit neuen Einträgen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_old = Path(tmpdir) / "old.csv"
            csv_new = Path(tmpdir) / "new.csv"

            df_old = pd.DataFrame({'="FKZ"': ["ABC001", "ABC002"]})
            df_new = pd.DataFrame({'="FKZ"': ["ABC001", "ABC002", "ABC003", "ABC004"]})

            df_old.to_csv(csv_old, sep=";", index=False, encoding="latin1")
            df_new.to_csv(csv_new, sep=";", index=False, encoding="latin1")

            stats = compare_csv_files(csv_old, csv_new)

            assert stats["added_count"] == 2
            assert stats["removed_count"] == 0
            assert "ABC003" in stats["added_fkz"]
            assert "ABC004" in stats["added_fkz"]

    def test_compare_csv_files_with_removals(self):
        """Test: CSV mit entfernten Einträgen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_old = Path(tmpdir) / "old.csv"
            csv_new = Path(tmpdir) / "new.csv"

            df_old = pd.DataFrame({'="FKZ"': ["ABC001", "ABC002", "ABC003"]})
            df_new = pd.DataFrame({'="FKZ"': ["ABC001"]})

            df_old.to_csv(csv_old, sep=";", index=False, encoding="latin1")
            df_new.to_csv(csv_new, sep=";", index=False, encoding="latin1")

            stats = compare_csv_files(csv_old, csv_new)

            assert stats["added_count"] == 0
            assert stats["removed_count"] == 2
            assert "ABC002" in stats["removed_fkz"]
            assert "ABC003" in stats["removed_fkz"]


class TestValidateWithFKZ:
    """Tests für validate_with_fkz() Convenience-Funktion."""

    def test_validate_with_fkz_returns_tuple(self, mock_faiss_empty, sample_df):
        """Test: Funktion gibt Tuple zurück."""
        result = validate_with_fkz(mock_faiss_empty, sample_df)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_validate_with_fkz_incomplete_index(self, mock_faiss_with_data, sample_df):
        """Test: Unvollständiger Index wird erkannt."""
        is_complete, new_count = validate_with_fkz(mock_faiss_with_data, sample_df)

        assert is_complete is False
        assert new_count == 2


class TestGetProjectsToIndex:
    """Tests für get_projects_to_index() Convenience-Funktion."""

    def test_get_projects_to_index_returns_dataframe(self, mock_faiss_with_data, sample_df):
        """Test: DataFrame wird zurückgegeben."""
        result = get_projects_to_index(mock_faiss_with_data, sample_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_get_projects_to_index_with_batch_size(self, mock_faiss_with_data, sample_df):
        """Test: Batch-Size begrenzt Ergebnis."""
        result = get_projects_to_index(mock_faiss_with_data, sample_df, batch_size=1)

        assert len(result) == 1


class TestGetNewProjectsSummary:
    """Tests für get_new_projects_summary()."""

    def test_summary_no_new_projects(self, sample_df):
        """Test: Zusammenfassung bei keinen neuen Projekten."""
        mock_faiss = MagicMock(spec=FaissStore)
        mock_faiss.id_map = {str(i): str(i) for i in range(5)}

        validator = FKZIndexValidator(mock_faiss, sample_df)
        summary = validator.get_new_projects_summary()

        assert "Keine neuen Projekte" in summary

    def test_summary_with_new_projects(self, mock_faiss_with_data, sample_df):
        """Test: Zusammenfassung mit neuen Projekten."""
        validator = FKZIndexValidator(mock_faiss_with_data, sample_df)
        summary = validator.get_new_projects_summary(limit=5)

        assert "2 neue Projekte" in summary
        assert "ABC004" in summary
        assert "ABC005" in summary

    def test_summary_truncates_long_text(self, mock_faiss_empty):
        """Test: Lange Strings werden gekürzt."""
        df = pd.DataFrame(
            {
                '="FKZ"': ["TEST001"],
                '="Zuwendungsempfänger"': ["X" * 100],  # Sehr lang
                '="Thema"': ["Y" * 100],  # Sehr lang
            }
        )

        validator = FKZIndexValidator(mock_faiss_empty, df)
        summary = validator.get_new_projects_summary()

        # Überprüfe dass Strings gekürzt wurden
        assert "..." in summary

    def test_summary_shows_continuation_message(self, mock_faiss_empty):
        """Test: Fortsetzungsmeldung bei mehr Projekten als Limit."""
        df = pd.DataFrame({'="FKZ"': [f"FKZ{i:03d}" for i in range(10)]})

        validator = FKZIndexValidator(mock_faiss_empty, df)
        summary = validator.get_new_projects_summary(limit=3)

        assert "und 7 weitere Projekte" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
