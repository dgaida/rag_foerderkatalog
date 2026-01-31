"""Unit-Tests für Hilfsfunktionen in src/app.py"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.app import extract_fkz_from_text, get_project_details, hybrid_rank, keyword_search, make_fkz_clickable


def test_extract_fkz_from_text():
    text = "Die Projekte FKZ: 13BDB60030 und FKZ 01AB123456 sind wichtig."
    result = extract_fkz_from_text(text)
    assert result == ["13BDB60030", "01AB123456"]


def test_extract_fkz_from_text_no_matches():
    text = "Keine FKZs hier."
    result = extract_fkz_from_text(text)
    assert result == []


def test_make_fkz_clickable():
    answer = "Hier ist das Projekt **FKZ: 13BDB60030**."
    result = make_fkz_clickable(answer)
    assert "**FKZ: [13BDB60030](#13BDB60030)**" in result


def test_keyword_search():
    df = pd.DataFrame(
        {
            "Thema": ["KI in der Medizin", "Robotik in der Fabrik", "Energieeffizienz"],
            "Zuwendungsempfänger": ["Uni A", "Institut B", "Firma C"],
        }
    )
    # Search for "KI"
    result = keyword_search(df, "KI", top_n=5)
    assert len(result) == 1
    assert "KI in der Medizin" in result["Thema"].values
    assert "__kw_score" in result.columns


def test_keyword_search_no_results():
    df = pd.DataFrame({"Thema": ["Test"]})
    result = keyword_search(df, "Unbekannt")
    assert result.empty


def test_hybrid_rank():
    sem_df = pd.DataFrame({"Thema": ["Semantic 1"]}, index=[10])
    sem_df["__score"] = [0.9]
    kw_df = pd.DataFrame({"Thema": ["Keyword 1"]}, index=[20])
    kw_df["__kw_score"] = [10]

    # Combined search
    result = hybrid_rank(sem_df, kw_df, k=10)
    assert len(result) == 2
    assert result.index[0] == 10  # Semantic first
    assert result.index[1] == 20


def test_hybrid_rank_deduplication():
    sem_df = pd.DataFrame({"Thema": ["Both"]}, index=[10])
    sem_df["__score"] = [0.9]
    kw_df = pd.DataFrame({"Thema": ["Both"]}, index=[10])
    kw_df["__kw_score"] = [10]

    result = hybrid_rank(sem_df, kw_df, k=10)
    assert len(result) == 1


def test_get_project_details():
    engine = MagicMock()
    engine.df = pd.DataFrame(
        {
            '="FKZ"': ["ABC12345"],
            '="Thema"': ["Test Thema"],
            '="Zuwendungsempfänger"': ["Test Empfänger"],
            '="Fördersumme in EUR"': [1000.0],
            "__laufzeit": ["2020-2022"],
        }
    )

    result = get_project_details("ABC12345", engine)
    assert "Test Thema" in result
    assert "Test Empfänger" in result
    assert "1,000.00 €" in result


def test_get_project_details_not_found():
    engine = MagicMock()
    engine.df = pd.DataFrame({'="FKZ"': ["OTHER"]})
    result = get_project_details("ABC12345", engine)
    assert "Kein Projekt mit FKZ" in result
