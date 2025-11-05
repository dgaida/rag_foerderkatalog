"""Unit-Tests für src/utils/index_validator.py

Tests für:
- Index-Validierung
- Identifikation fehlender Einträge
- Identifikation verwaister Einträge
- Statistik-Generierung
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.utils.index_validator import (
    IndexValidator,
    check_index_completeness,
    get_new_projects_summary,
)
from src.embeddings.faiss_store import FaissStore


# ===== Fixtures auf Modul-Ebene (verfügbar für alle Testklassen) =====


@pytest.fixture
def sample_df():
    """Erstellt einen Test-DataFrame."""
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
def mock_faiss_partial():
    """Erstellt einen teilweise gefüllten Mock-FAISS-Store."""
    mock = MagicMock(spec=FaissStore)
    mock.index = MagicMock()
    mock.index.ntotal = 3
    # Nur IDs 0, 1, 2 sind indiziert
    mock.id_map = {"0": "0", "1": "1", "2": "2"}
    return mock


@pytest.fixture
def mock_faiss_complete():
    """Erstellt einen vollständig gefüllten Mock-FAISS-Store."""
    mock = MagicMock(spec=FaissStore)
    mock.index = MagicMock()
    mock.index.ntotal = 5
    mock.id_map = {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"}
    return mock


@pytest.fixture
def mock_faiss_with_orphans():
    """Erstellt einen Mock-FAISS-Store mit verwaisten Einträgen."""
    mock = MagicMock(spec=FaissStore)
    mock.index = MagicMock()
    mock.index.ntotal = 6
    # IDs 5 und 6 existieren nicht im DataFrame
    mock.id_map = {"0": "0", "1": "1", "2": "2", "3": "5", "4": "6"}  # Verwaist  # Verwaist
    return mock


# ===== Test-Klassen =====


class TestGetIndexedIds:
    """Tests für get_indexed_ids()."""

    def test_get_indexed_ids_empty_index(self, mock_faiss_empty, sample_df):
        """Test: Leerer Index liefert leere Menge."""
        validator = IndexValidator(mock_faiss_empty, sample_df)
        result = validator.get_indexed_ids()

        assert result == set()

    def test_get_indexed_ids_partial_index(self, mock_faiss_partial, sample_df):
        """Test: Teilweise gefüllter Index liefert korrekte IDs."""
        validator = IndexValidator(mock_faiss_partial, sample_df)
        result = validator.get_indexed_ids()

        assert result == {0, 1, 2}

    def test_get_indexed_ids_complete_index(self, mock_faiss_complete, sample_df):
        """Test: Vollständiger Index liefert alle IDs."""
        validator = IndexValidator(mock_faiss_complete, sample_df)
        result = validator.get_indexed_ids()

        assert result == {0, 1, 2, 3, 4}

    def test_get_indexed_ids_with_invalid_ids(self, sample_df):
        """Test: Ungültige IDs werden übersprungen."""
        mock_faiss = MagicMock(spec=FaissStore)
        mock_faiss.id_map = {"0": "0", "1": "invalid", "2": "2"}

        validator = IndexValidator(mock_faiss, sample_df)
        result = validator.get_indexed_ids()

        # "invalid" sollte übersprungen werden
        assert result == {0, 2}


class TestGetCsvIds:
    """Tests für get_csv_ids()."""

    def test_get_csv_ids_normal_dataframe(self, mock_faiss_empty, sample_df):
        """Test: DataFrame-IDs werden korrekt extrahiert."""
        validator = IndexValidator(mock_faiss_empty, sample_df)
        result = validator.get_csv_ids()

        assert result == {0, 1, 2, 3, 4}

    def test_get_csv_ids_empty_dataframe(self, mock_faiss_empty):
        """Test: Leerer DataFrame liefert leere Menge."""
        empty_df = pd.DataFrame()
        validator = IndexValidator(mock_faiss_empty, empty_df)
        result = validator.get_csv_ids()

        assert result == set()

    def test_get_csv_ids_custom_index(self, mock_faiss_empty):
        """Test: Custom DataFrame-Index wird korrekt gehandhabt."""
        df = pd.DataFrame({"data": [1, 2, 3]}, index=[10, 20, 30])
        validator = IndexValidator(mock_faiss_empty, df)
        result = validator.get_csv_ids()

        assert result == {10, 20, 30}


class TestGetMissingIndices:
    """Tests für get_missing_indices()."""

    def test_get_missing_indices_empty_index(self, mock_faiss_empty, sample_df):
        """Test: Leerer Index bedeutet alle IDs fehlen."""
        validator = IndexValidator(mock_faiss_empty, sample_df)
        result = validator.get_missing_indices()

        assert result == [0, 1, 2, 3, 4]

    def test_get_missing_indices_partial_index(self, mock_faiss_partial, sample_df):
        """Test: Teilweise gefüllter Index identifiziert fehlende IDs."""
        validator = IndexValidator(mock_faiss_partial, sample_df)
        result = validator.get_missing_indices()

        assert result == [3, 4]

    def test_get_missing_indices_complete_index(self, mock_faiss_complete, sample_df):
        """Test: Vollständiger Index hat keine fehlenden IDs."""
        validator = IndexValidator(mock_faiss_complete, sample_df)
        result = validator.get_missing_indices()

        assert result == []

    def test_get_missing_indices_sorted(self, mock_faiss_partial, sample_df):
        """Test: Ergebnis ist sortiert."""
        validator = IndexValidator(mock_faiss_partial, sample_df)
        result = validator.get_missing_indices()

        assert result == sorted(result)


class TestGetOrphanedIndices:
    """Tests für get_orphaned_indices()."""

    def test_get_orphaned_indices_no_orphans(self, mock_faiss_complete, sample_df):
        """Test: Keine verwaisten Einträge."""
        validator = IndexValidator(mock_faiss_complete, sample_df)
        result = validator.get_orphaned_indices()

        assert result == []

    def test_get_orphaned_indices_with_orphans(self, mock_faiss_with_orphans, sample_df):
        """Test: Verwaiste Einträge werden identifiziert."""
        validator = IndexValidator(mock_faiss_with_orphans, sample_df)
        result = validator.get_orphaned_indices()

        assert result == [5, 6]

    def test_get_orphaned_indices_empty_csv(self, mock_faiss_complete):
        """Test: Alle Index-Einträge sind verwaist wenn CSV leer."""
        empty_df = pd.DataFrame()
        validator = IndexValidator(mock_faiss_complete, empty_df)
        result = validator.get_orphaned_indices()

        assert result == [0, 1, 2, 3, 4]


class TestValidateIndex:
    """Tests für validate_index()."""

    def test_validate_index_empty(self, mock_faiss_empty, sample_df):
        """Test: Leerer Index liefert is_valid=False."""
        validator = IndexValidator(mock_faiss_empty, sample_df)
        is_valid, stats = validator.validate_index()

        assert is_valid is False
        assert stats["is_empty"] is True
        assert stats["csv_total"] == 5
        assert stats["indexed_total"] == 0
        assert stats["missing_count"] == 5
        assert stats["sync_percentage"] == 0.0

    def test_validate_index_partial(self, mock_faiss_partial, sample_df):
        """Test: Teilweise gefüllter Index."""
        validator = IndexValidator(mock_faiss_partial, sample_df)
        is_valid, stats = validator.validate_index()

        assert is_valid is False
        assert stats["is_empty"] is False
        assert stats["csv_total"] == 5
        assert stats["indexed_total"] == 3
        assert stats["missing_count"] == 2
        assert stats["sync_percentage"] == 60.0  # 3/5 = 60%

    def test_validate_index_complete(self, mock_faiss_complete, sample_df):
        """Test: Vollständiger Index liefert is_valid=True."""
        validator = IndexValidator(mock_faiss_complete, sample_df)
        is_valid, stats = validator.validate_index()

        assert is_valid is True
        assert stats["is_empty"] is False
        assert stats["csv_total"] == 5
        assert stats["indexed_total"] == 5
        assert stats["missing_count"] == 0
        assert stats["sync_percentage"] == 100.0

    def test_validate_index_with_orphans(self, mock_faiss_with_orphans, sample_df):
        """Test: Verwaiste Einträge werden in Statistik erfasst."""
        validator = IndexValidator(mock_faiss_with_orphans, sample_df)
        is_valid, stats = validator.validate_index()

        assert stats["orphaned_count"] == 2
        assert len(stats["orphaned_ids"]) > 0


class TestGetMissingProjects:
    """Tests für get_missing_projects()."""

    def test_get_missing_projects_no_missing(self, mock_faiss_complete, sample_df):
        """Test: Keine fehlenden Projekte liefert leeres DataFrame."""
        validator = IndexValidator(mock_faiss_complete, sample_df)
        result = validator.get_missing_projects()

        assert result.empty

    def test_get_missing_projects_with_missing(self, mock_faiss_partial, sample_df):
        """Test: Fehlende Projekte werden als DataFrame zurückgegeben."""
        validator = IndexValidator(mock_faiss_partial, sample_df)
        result = validator.get_missing_projects()

        assert len(result) == 2
        assert 3 in result.index
        assert 4 in result.index
        assert '="FKZ"' in result.columns

    def test_get_missing_projects_with_limit(self, mock_faiss_partial, sample_df):
        """Test: Limit begrenzt Anzahl der zurückgegebenen Projekte."""
        validator = IndexValidator(mock_faiss_partial, sample_df)
        result = validator.get_missing_projects(limit=1)

        assert len(result) == 1
        assert 3 in result.index


class TestCheckIndexCompleteness:
    """Tests für check_index_completeness()."""

    def test_check_completeness_empty(self, mock_faiss_empty, sample_df):
        """Test: Leerer Index wird erkannt."""
        is_complete, missing = check_index_completeness(mock_faiss_empty, sample_df)

        assert is_complete is False
        assert missing == 5

    def test_check_completeness_partial(self, mock_faiss_partial, sample_df):
        """Test: Teilweise gefüllter Index."""
        is_complete, missing = check_index_completeness(mock_faiss_partial, sample_df)

        assert is_complete is False
        assert missing == 2

    def test_check_completeness_complete(self, mock_faiss_complete, sample_df):
        """Test: Vollständiger Index."""
        is_complete, missing = check_index_completeness(mock_faiss_complete, sample_df)

        assert is_complete is True
        assert missing == 0


class TestGetNewProjectsSummary:
    """Tests für get_new_projects_summary()."""

    def test_summary_no_missing(self, mock_faiss_complete, sample_df):
        """Test: Keine fehlenden Projekte."""
        validator = IndexValidator(mock_faiss_complete, sample_df)
        summary = get_new_projects_summary(validator)

        assert "Keine neuen Projekte" in summary

    def test_summary_with_missing(self, mock_faiss_partial, sample_df):
        """Test: Zusammenfassung mit fehlenden Projekten."""
        validator = IndexValidator(mock_faiss_partial, sample_df)
        summary = get_new_projects_summary(validator)

        assert "2 neue Projekte" in summary
        assert "ABC004" in summary or "ABC005" in summary

    def test_summary_truncates_long_text(self):
        """Test: Lange Texte werden gekürzt."""
        # Erstelle DataFrame mit langen Strings
        df = pd.DataFrame(
            {
                '="FKZ"': ["TEST001"],
                '="Zuwendungsempfänger"': ["X" * 100],
                '="Thema"': ["Y" * 100],
            }  # Sehr lang  # Sehr lang
        )

        mock_faiss = MagicMock(spec=FaissStore)
        mock_faiss.index = None
        mock_faiss.id_map = {}

        validator = IndexValidator(mock_faiss, df)
        summary = get_new_projects_summary(validator)

        # Überprüfe dass Strings gekürzt wurden
        assert "..." in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
