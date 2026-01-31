"""Unit-Tests für src/utils/index_validator.py"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.embeddings.faiss_store import FaissStore
from src.utils.index_validator import IndexValidator, check_index_completeness, get_new_projects_summary


@pytest.fixture
def sample_df():
    return pd.DataFrame({"Data": [1, 2, 3, 4, 5]}, index=[0, 1, 2, 3, 4])


@pytest.fixture
def mock_faiss():
    mock = MagicMock(spec=FaissStore)
    mock.index = MagicMock()
    mock.index.ntotal = 3
    mock.id_map = {"0": "0", "1": "1", "2": "2"}
    return mock


class TestIndexValidator:
    def test_get_indexed_ids(self, mock_faiss, sample_df):
        validator = IndexValidator(mock_faiss, sample_df)
        indexed_ids = validator.get_indexed_ids()
        assert indexed_ids == {0, 1, 2}

    def test_get_indexed_ids_invalid(self, sample_df):
        mock = MagicMock(spec=FaissStore)
        mock.id_map = {"0": "0", "1": "invalid"}
        validator = IndexValidator(mock, sample_df)
        indexed_ids = validator.get_indexed_ids()
        assert indexed_ids == {0}

    def test_get_csv_ids(self, mock_faiss, sample_df):
        validator = IndexValidator(mock_faiss, sample_df)
        csv_ids = validator.get_csv_ids()
        assert csv_ids == {0, 1, 2, 3, 4}

    def test_get_missing_indices(self, mock_faiss, sample_df):
        validator = IndexValidator(mock_faiss, sample_df)
        missing = validator.get_missing_indices()
        assert missing == [3, 4]

    def test_get_orphaned_indices(self, mock_faiss, sample_df):
        # Index has more than CSV
        mock_faiss.id_map["5"] = "5"
        validator = IndexValidator(mock_faiss, sample_df)
        orphaned = validator.get_orphaned_indices()
        assert orphaned == [5]

    def test_validate_index_empty(self, sample_df):
        mock = MagicMock(spec=FaissStore)
        mock.index = None
        mock.id_map = {}
        validator = IndexValidator(mock, sample_df)
        is_valid, stats = validator.validate_index()
        assert is_valid is False
        assert stats["is_empty"] is True

    def test_validate_index_partial(self, mock_faiss, sample_df):
        validator = IndexValidator(mock_faiss, sample_df)
        is_valid, stats = validator.validate_index()
        assert is_valid is False
        assert stats["missing_count"] == 2
        assert stats["sync_percentage"] == 60.0

    def test_get_missing_projects(self, mock_faiss, sample_df):
        validator = IndexValidator(mock_faiss, sample_df)
        missing_df = validator.get_missing_projects(limit=1)
        assert len(missing_df) == 1
        assert missing_df.index[0] == 3


def test_check_index_completeness(mock_faiss, sample_df):
    is_complete, missing_count = check_index_completeness(mock_faiss, sample_df)
    assert is_complete is False
    assert missing_count == 2


def test_get_new_projects_summary(mock_faiss, sample_df):
    # Add required columns for summary
    sample_df['="FKZ"'] = ["F1", "F2", "F3", "F4", "F5"]
    sample_df['="Zuwendungsempfänger"'] = ["E1", "E2", "E3", "E4", "E5"]
    sample_df['="Thema"'] = ["T1", "T2", "T3", "T4", "T5"]

    validator = IndexValidator(mock_faiss, sample_df)
    summary = get_new_projects_summary(validator)
    assert "2 neue Projekte" in summary
    assert "F4" in summary
    assert "F5" in summary


def test_get_new_projects_summary_empty(sample_df):
    mock = MagicMock(spec=FaissStore)
    mock.id_map = {str(i): str(i) for i in range(5)}
    validator = IndexValidator(mock, sample_df)
    summary = get_new_projects_summary(validator)
    assert "Keine neuen Projekte" in summary
