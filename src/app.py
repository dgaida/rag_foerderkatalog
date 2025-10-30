#!/usr/bin/env python3
"""
src/app.py

Gradio-Oberfläche für die hybride Suche (semantisch + keyword).
Die hybride Suche kombiniert:
- Semantische Suche (FAISS) — liefert beliebig viele Treffer (config: k)
- Keyword-Suche (Pandas simple matching) — Top-5 Treffer
Die finalen Treffer sind eine deduplizierte Kombination (semantische Treffer zuerst, dann keyword).
"""
from __future__ import annotations

from typing import Optional, List
import pandas as pd
import gradio as gr
import logging

from .utils.logging_config import get_logger
from .search.engine import ProjectSearchEngine
from .config import TOP_K_DEFAULT

logger = get_logger(__name__)


def keyword_search(df: pd.DataFrame, query: str, top_n: int = 5, text_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Einfache keyword-/substring-Suche über ausgewählte Textspalten.
    Bewertet Ergebnisse nach Anzahl der Treffer (Count) und gibt top_n zurück.

    Args:
        df: DataFrame mit Projektdaten.
        query: Nutzeranfrage (String).
        top_n: Anzahl der zurückzugebenden Treffer (Default: 5).
        text_columns: Liste von Spalten, die durchsucht werden. Wenn None, werden gängige Spalten genutzt.

    Returns:
        DataFrame mit top_n Treffern (inkl. Spalte '__kw_score').
    """
    if df is None or df.empty:
        return pd.DataFrame()

    qc = query.strip().lower()
    if not qc:
        return pd.DataFrame()

    # Auswahl der Spalten
    text_columns = text_columns or [c for c in ("Zuwendungsempfänger", "Zuwendungsempf\u00e4nger", "Thema", "Klartext Leistungsplansystematik", "Ausf\u00fchrende Stelle") if c in df.columns]

    scores = []
    for idx, row in df.iterrows():
        score = 0
        for col in text_columns:
            val = str(row.get(col, "")).lower()
            if not val:
                continue
            # weight: occurrences of whole query and individual token matches
            if qc in val:
                score += 10
            for token in qc.split():
                if token in val:
                    score += 1
        if score > 0:
            scores.append((idx, score))

    if not scores:
        return pd.DataFrame()

    # Top-N nach Score
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    indices = [idx for idx, _ in scores]
    result = df.loc[indices].copy()
    result["__kw_score"] = [s for _, s in scores]
    return result


def hybrid_rank(sem_df: pd.DataFrame, kw_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Kombiniert semantische Ergebnisse (sem_df) und keyword Ergebnisse (kw_df),
    priorisiert semantische Treffer und hängt keyword-Treffer an, ohne Duplikate.

    Args:
        sem_df: DataFrame der semantischen Treffer (mit '__score' Spalte).
        kw_df: DataFrame der keyword Treffer (mit '__kw_score' Spalte).
        k: gewünschte Gesamtanzahl der finalen Treffer.

    Returns:
        DataFrame mit maximal k Treffern.
    """
    if sem_df is None:
        sem_df = pd.DataFrame()
    if kw_df is None:
        kw_df = pd.DataFrame()

    # Beginne mit semantischen Treffern (nach Score sortiert)
    sem_df_sorted = sem_df.sort_values("__score", ascending=False) if "__score" in sem_df.columns else sem_df
    combined = []
    added_ids = set()

    # Add semantic first
    for idx, row in sem_df_sorted.iterrows():
        if len(combined) >= k:
            break
        if idx in added_ids:
            continue
        combined.append(row)
        added_ids.add(idx)

    # Then keyword results
    kw_df_sorted = kw_df.sort_values("__kw_score", ascending=False) if "__kw_score" in kw_df.columns else kw_df
    for idx, row in kw_df_sorted.iterrows():
        if len(combined) >= k:
            break
        if idx in added_ids:
            continue
        combined.append(row)
        added_ids.add(idx)

    if not combined:
        return pd.DataFrame()

    result = pd.DataFrame(combined)
    return result


def build_ui(engine: ProjectSearchEngine) -> gr.Blocks:
    """
    Baut die Gradio-Benutzeroberfläche und bindet die Suchfunktionen an die Widgets.
    """
    with gr.Blocks(title="Förderprojekte RAG — Hybride Suche") as demo:
        gr.Markdown("## Förderprojekte — Hybride Suche (Semantisch + Keyword)")
        with gr.Row():
            query_input = gr.Textbox(label="Suchanfrage", placeholder="z. B. 'KI Projekte Niedersachsen Universität' ...", lines=1)
            mode = gr.Radio(choices=["hybrid", "semantic", "keyword"], value="hybrid", label="Suchmodus")
            k_slider = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Anzahl semantische Treffer (k)")
            search_btn = gr.Button("Suchen")

        result_table = gr.Dataframe(headers=None, interactive=False, label="Treffer")
        llm_answer = gr.Textbox(label="LLM-Antwort (kontextbasiert)", interactive=False)

        def on_search(query: str, mode_choice: str, k: int):
            logger.info("Suchanfrage: mode=%s, k=%s, query=%s", mode_choice, k, query)
            if engine.df is None:
                return pd.DataFrame(), "DataFrame nicht geladen."

            sem_df = pd.DataFrame()
            kw_df = pd.DataFrame()

            if mode_choice in ("hybrid", "semantic"):
                sem_df = engine.search(query, k=int(k))
            if mode_choice in ("hybrid", "keyword"):
                # Keyword search über die ganze Tabelle, we take top 5
                kw_df = keyword_search(engine.df, query, top_n=5)

            if mode_choice == "semantic":
                final = sem_df.head(int(k)) if not sem_df.empty else pd.DataFrame()
            elif mode_choice == "keyword":
                final = kw_df
            else:  # hybrid
                final = hybrid_rank(sem_df, kw_df, int(k))

            # Anzeige: wähle sinnvolle Spalten
            if final is None or final.empty:
                return pd.DataFrame(), "Keine Treffer."

            display_cols = [c for c in ("FKZ", "Zuwendungsempfänger", "Zuwendungsempf\u00e4nger", "Thema", "F\u00f6rdersumme in EUR", "__score", "__kw_score") if c in final.columns]
            display_df = final[display_cols].copy()

            # LLM Antwort (kontextbasiert) nur, wenn semantische Informationen vorliegen
            try:
                answer = engine.answer_with_context(query)
            except Exception as e:
                logger.exception("Fehler beim LLM-Abruf: %s", e)
                answer = "LLM Antwort konnte nicht erzeugt werden."

            return display_df, answer

        search_btn.click(on_search, inputs=[query_input, mode, k_slider], outputs=[result_table, llm_answer])

    return demo


# Für den direkten Start via `python -m src.app` falls gewünscht
if __name__ == "__main__":
    setup_logging()
    logger.info("Starte Gradio-App (src.app) standalone")
    engine = ProjectSearchEngine()
    engine.load_and_clean()
    # Beim direkten Start: nur dann Embeddings erzeugen, wenn Index leer
    try:
        engine.build_embeddings_if_missing()
    except Exception:
        logger.exception("Fehler beim Erzeugen der Embeddings beim App-Start")
    demo = build_ui(engine)
    demo.launch()
