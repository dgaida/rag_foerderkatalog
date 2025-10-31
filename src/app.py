#!/usr/bin/env python3
"""
src/app.py

Gradio-Oberfläche für die hybride Suche (semantisch + keyword) mit modernem Design.
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

from .utils.logging_config import get_logger, setup_logging
from .search.engine import ProjectSearchEngine
from .config import TOP_K_DEFAULT

logger = get_logger(__name__)

# Modernes CSS-Styling
CUSTOM_CSS = """
/* ===== Globale Variablen & Theme ===== */
:root {
    --primary-color: #4f46e5;
    --primary-hover: #4338ca;
    --secondary-color: #06b6d4;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --danger-color: #ef4444;
    --dark-bg: #0f172a;
    --card-bg: #1e293b;
    --border-color: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    --border-radius: 12px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ===== Body & Container ===== */
body {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: var(--text-primary);
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}

/* ===== Header Styling ===== */
.markdown-header {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    padding: 2.5rem 2rem;
    border-radius: var(--border-radius);
    margin-bottom: 2rem;
    box-shadow: var(--shadow-lg);
    text-align: center;
}

.markdown-header h2 {
    color: white !important;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    margin: 0 0 0.5rem 0 !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.markdown-header p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.1rem !important;
    margin: 0 !important;
}

/* ===== Cards & Blocks ===== */
.block {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--border-radius) !important;
    padding: 1.5rem !important;
    box-shadow: var(--shadow-md) !important;
    transition: var(--transition) !important;
}

.block:hover {
    box-shadow: var(--shadow-lg) !important;
    transform: translateY(-2px);
}

/* ===== Input Fields ===== */
.input-container label {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.5rem !important;
    display: block;
}

input[type="text"],
textarea {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    padding: 0.75rem 1rem !important;
    font-size: 1rem !important;
    transition: var(--transition) !important;
}

input[type="text"]:focus,
textarea:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
    outline: none !important;
}

input[type="text"]::placeholder,
textarea::placeholder {
    color: var(--text-secondary) !important;
}

/* ===== Buttons ===== */
.primary-button,
button[type="submit"] {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-hover) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    cursor: pointer !important;
    transition: var(--transition) !important;
    box-shadow: var(--shadow-md) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.primary-button:hover,
button[type="submit"]:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg) !important;
}

.primary-button:active,
button[type="submit"]:active {
    transform: translateY(0);
}

/* ===== Radio Buttons ===== */
.radio-group {
    background: rgba(15, 23, 42, 0.4) !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
}

.radio-group label {
    color: var(--text-primary) !important;
    padding: 0.5rem 1rem !important;
    border-radius: 6px !important;
    transition: var(--transition) !important;
    cursor: pointer;
}

.radio-group input[type="radio"]:checked + label {
    background: var(--primary-color) !important;
    color: white !important;
}

.radio-group label:hover {
    background: rgba(79, 70, 229, 0.2) !important;
}

/* ===== Slider ===== */
.slider-container {
    padding: 1rem 0 !important;
}

input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 8px;
    border-radius: 4px;
    background: var(--border-color);
    outline: none;
}

input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: var(--primary-color);
    cursor: pointer;
    box-shadow: var(--shadow-md);
    transition: var(--transition);
}

input[type="range"]::-webkit-slider-thumb:hover {
    background: var(--primary-hover);
    transform: scale(1.2);
}

input[type="range"]::-moz-range-thumb {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: var(--primary-color);
    cursor: pointer;
    border: none;
    box-shadow: var(--shadow-md);
}

/* ===== DataTable ===== */
.dataframe {
    background: var(--dark-bg) !important;
    border-radius: var(--border-radius) !important;
    overflow: hidden;
    box-shadow: var(--shadow-md) !important;
}

.dataframe thead {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
}

.dataframe thead th {
    color: white !important;
    font-weight: 600 !important;
    padding: 1rem !important;
    text-align: left !important;
    border-bottom: 2px solid var(--border-color) !important;
}

.dataframe tbody tr {
    border-bottom: 1px solid var(--border-color) !important;
    transition: var(--transition) !important;
}

.dataframe tbody tr:hover {
    background: rgba(79, 70, 229, 0.1) !important;
}

.dataframe tbody td {
    color: var(--text-primary) !important;
    padding: 0.75rem 1rem !important;
}

/* ===== Textbox (LLM Answer) ===== */
.output-textbox {
    background: var(--dark-bg) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: var(--border-radius) !important;
    color: var(--text-primary) !important;
    padding: 1.5rem !important;
    font-size: 1rem !important;
    line-height: 1.8 !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: var(--shadow-md) !important;
}

/* ===== Tabs ===== */
.tabs {
    border-bottom: 2px solid var(--border-color) !important;
    margin-bottom: 1.5rem !important;
}

.tab-nav button {
    color: var(--text-secondary) !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    padding: 1rem 1.5rem !important;
    font-weight: 600 !important;
    transition: var(--transition) !important;
}

.tab-nav button:hover {
    color: var(--text-primary) !important;
    background: rgba(79, 70, 229, 0.1) !important;
}

.tab-nav button.selected {
    color: var(--primary-color) !important;
    border-bottom-color: var(--primary-color) !important;
}

/* ===== Loading Animation ===== */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid var(--border-color);
    border-radius: 50%;
    border-top-color: var(--primary-color);
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* ===== Info Cards ===== */
.info-card {
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%);
    border-left: 4px solid var(--primary-color);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}

.info-card-title {
    color: var(--primary-color);
    font-weight: 700;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.info-card-content {
    color: var(--text-primary);
    font-size: 0.95rem;
    line-height: 1.6;
}

/* ===== Responsive Design ===== */
@media (max-width: 768px) {
    .gradio-container {
        padding: 1rem !important;
    }

    .markdown-header {
        padding: 1.5rem 1rem;
    }

    .markdown-header h2 {
        font-size: 1.8rem !important;
    }

    .block {
        padding: 1rem !important;
    }
}

/* ===== Scrollbar Styling ===== */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: var(--dark-bg);
}

::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary-color);
}

/* ===== Animations ===== */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.block {
    animation: fadeIn 0.5s ease-out;
}

/* ===== Status Badges ===== */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
}

.status-badge.success {
    background: rgba(16, 185, 129, 0.2);
    color: var(--success-color);
}

.status-badge.warning {
    background: rgba(245, 158, 11, 0.2);
    color: var(--warning-color);
}

.status-badge.info {
    background: rgba(6, 182, 212, 0.2);
    color: var(--secondary-color);
}
"""


def keyword_search(
        df: pd.DataFrame,
        query: str,
        top_n: int = 5,
        text_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Einfache keyword-/substring-Suche über ausgewählte Textspalten.

    Bewertet Ergebnisse nach Anzahl der Treffer (Count) und gibt top_n zurück.

    Args:
        df: DataFrame mit Projektdaten.
        query: Nutzeranfrage (String).
        top_n: Anzahl der zurückzugebenden Treffer. Defaults to 5.
        text_columns: Liste von Spalten, die durchsucht werden.
            Wenn None, werden gängige Spalten genutzt.

    Returns:
        pd.DataFrame: DataFrame mit top_n Treffern (inkl. Spalte '__kw_score').

    Example:
        >>> results = keyword_search(df, "KI Bayern", top_n=5)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    qc = query.strip().lower()
    if not qc:
        return pd.DataFrame()

    # Auswahl der Spalten
    text_columns = text_columns or [
        c for c in (
            "Zuwendungsempfänger", "Zuwendungsempf\u00e4nger",
            "Thema", "Klartext Leistungsplansystematik",
            "Ausf\u00fchrende Stelle"
        ) if c in df.columns
    ]

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
    """Kombiniert semantische Ergebnisse (sem_df) und keyword Ergebnisse (kw_df).

    Priorisiert semantische Treffer und hängt keyword-Treffer an, ohne Duplikate.

    Args:
        sem_df: DataFrame der semantischen Treffer (mit '__score' Spalte).
        kw_df: DataFrame der keyword Treffer (mit '__kw_score' Spalte).
        k: Gewünschte Gesamtanzahl der finalen Treffer.

    Returns:
        pd.DataFrame: DataFrame mit maximal k Treffern.
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
    """Baut die Gradio-Benutzeroberfläche mit modernem Design.

    Args:
        engine: Initialisierte ProjectSearchEngine-Instanz.

    Returns:
        gr.Blocks: Konfigurierte Gradio-App.
    """
    with gr.Blocks(
            title="🧠 RAG Förderkatalog — Intelligente Projektsuche",
            css=CUSTOM_CSS,
            theme=gr.themes.Default(
                primary_hue="indigo",
                secondary_hue="cyan",
                neutral_hue="slate",
                font=("Inter", "system-ui", "sans-serif")
            )
    ) as demo:

        # Header
        gr.Markdown(
            """
            ## 🧠 RAG Förderkatalog
            **Intelligente Suche in deutschen Forschungsförderprojekten**
            """,
            elem_classes="markdown-header"
        )

        # Info Card
        with gr.Row():
            gr.Markdown(
                """
                <div class="info-card">
                    <div class="info-card-title">💡 Über diese Anwendung</div>
                    <div class="info-card-content">
                        Diese KI-gestützte Suchmaschine durchsucht über 200.000 Förderprojekte des BMBF 
                        mit semantischer Vektorsuche und kontextbasierter KI-Antwortgenerierung.
                        Wählen Sie zwischen <strong>Semantischer</strong>, <strong>Keyword</strong> oder 
                        <strong>Hybrider Suche</strong> für optimale Ergebnisse.
                    </div>
                </div>
                """
            )

        # Search Controls
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="🔍 Suchanfrage",
                    placeholder="z. B. 'Künstliche Intelligenz Projekte Bayern 2020-2025' oder 'Quantencomputing Hochschulen'",
                    lines=2,
                    elem_classes="input-container"
                )

        with gr.Row():
            with gr.Column(scale=2):
                mode = gr.Radio(
                    choices=["hybrid", "semantic", "keyword"],
                    value="hybrid",
                    label="⚙️ Suchmodus",
                    elem_classes="radio-group"
                )
            with gr.Column(scale=2):
                k_slider = gr.Slider(
                    minimum=5,
                    maximum=100,
                    value=20,
                    step=5,
                    label="📊 Anzahl Treffer (k)",
                    elem_classes="slider-container"
                )
            with gr.Column(scale=1):
                search_btn = gr.Button(
                    "🚀 Suchen",
                    variant="primary",
                    elem_classes="primary-button"
                )

        # Results Section
        gr.Markdown("---")

        with gr.Tabs():
            with gr.Tab("📋 Suchergebnisse"):
                result_table = gr.Dataframe(
                    headers=None,
                    interactive=False,
                    label="Gefundene Projekte",
                    wrap=True,
                    elem_classes="dataframe"
                )

                with gr.Accordion("📈 Statistiken", open=False):
                    stats_output = gr.Markdown()

            with gr.Tab("🤖 KI-Analyse"):
                llm_answer = gr.Textbox(
                    label="Kontextbasierte KI-Antwort",
                    interactive=False,
                    lines=15,
                    elem_classes="output-textbox"
                )

        # Examples Section
        gr.Markdown("### 💡 Beispielsuchen")
        gr.Examples(
            examples=[
                ["Künstliche Intelligenz Hochschule Bayern", "hybrid", 20],
                ["Wasserstoff Energie NRW 2020-2025", "semantic", 15],
                ["Quantencomputing Forschung", "semantic", 25],
                ["Klimawandel Digitalisierung", "hybrid", 30],
                ["Medizintechnik Berlin", "keyword", 10],
            ],
            inputs=[query_input, mode, k_slider],
            label="Klicken Sie auf ein Beispiel zum Ausprobieren"
        )

        # Search Function
        def on_search(query: str, mode_choice: str, k: int):
            """Führt die Suche durch und generiert Ergebnisse.

            Args:
                query: Suchanfrage.
                mode_choice: Gewählter Suchmodus (hybrid/semantic/keyword).
                k: Anzahl der gewünschten Treffer.

            Returns:
                Tuple: (DataFrame mit Ergebnissen, LLM-Antwort, Statistik-Markdown)
            """
            logger.info("Suchanfrage: mode=%s, k=%s, query=%s", mode_choice, k, query)

            if not query or not query.strip():
                return (
                    pd.DataFrame(),
                    "⚠️ Bitte geben Sie eine Suchanfrage ein.",
                    ""
                )

            if engine.df is None:
                return (
                    pd.DataFrame(),
                    "❌ DataFrame nicht geladen. Bitte starten Sie die Anwendung neu.",
                    ""
                )

            sem_df = pd.DataFrame()
            kw_df = pd.DataFrame()

            try:
                if mode_choice in ("hybrid", "semantic"):
                    sem_df = engine.search(query, k=int(k))
                if mode_choice in ("hybrid", "keyword"):
                    kw_df = keyword_search(engine.df, query, top_n=5)

                if mode_choice == "semantic":
                    final = sem_df.head(int(k)) if not sem_df.empty else pd.DataFrame()
                elif mode_choice == "keyword":
                    final = kw_df
                else:  # hybrid
                    final = hybrid_rank(sem_df, kw_df, int(k))

                if final is None or final.empty:
                    return (
                        pd.DataFrame(),
                        "🔍 Keine Treffer gefunden. Versuchen Sie andere Suchbegriffe.",
                        ""
                    )

                # Display columns
                display_cols = [
                    c for c in (
                        '="FKZ"', '="Zuwendungsempfänger"',
                        "Zuwendungsempf\u00e4nger", '="Thema"',
                        '="Bundesland"', '__laufzeit',
                        '="Fördersumme in EUR"',
                        "__score", "__kw_score"
                    ) if c in final.columns
                ]
                display_df = final[display_cols].copy()

                # Formatierung
                if '="Fördersumme in EUR"' in display_df.columns:
                    display_df['="Fördersumme in EUR"'] = display_df['="Fördersumme in EUR"'].apply(
                        lambda x: f"{x:,.2f} €" if pd.notna(x) else "N/A"
                    )

                if '__score' in display_df.columns:
                    display_df['__score'] = display_df['__score'].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) else ""
                    )

                # Statistiken
                total_results = len(final)
                total_funding = final['="Fördersumme in EUR"'].sum() if '="Fördersumme in EUR"' in final.columns else 0

                if isinstance(total_funding, str):
                    # Falls bereits formatiert wurde
                    total_funding = 0

                stats_md = f"""
                ### 📊 Suchstatistiken

                - **Treffer gefunden**: {total_results}
                - **Gesamtfördersumme**: {total_funding:,.2f} €
                - **Suchmodus**: {mode_choice.upper()}
                - **Query**: _{query}_
                """

                # LLM Antwort
                try:
                    answer = engine.answer_with_context(query)
                except Exception as e:
                    logger.exception("Fehler beim LLM-Abruf: %s", e)
                    answer = f"⚠️ LLM-Antwort konnte nicht generiert werden.\n\n**Fehler**: {str(e)}"

                return display_df, answer, stats_md

            except Exception as e:
                logger.exception("Fehler während der Suche: %s", e)
                return (
                    pd.DataFrame(),
                    f"❌ Ein Fehler ist aufgetreten: {str(e)}",
                    ""
                )

        search_btn.click(
            on_search,
            inputs=[query_input, mode, k_slider],
            outputs=[result_table, llm_answer, stats_output]
        )

        # Footer
        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: var(--text-secondary); font-size: 0.9rem;">
                <p>🚀 Powered by <strong>Ollama</strong>, <strong>FAISS</strong>, <strong>Gradio</strong> & <strong>LLMClient</strong></p>
                <p>📊 Datenquelle: BMBF Förderkatalog | 🧠 RAG-basierte semantische Suche</p>
            </div>
            """
        )

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
    demo.launch(share=False, inbrowser=True)
