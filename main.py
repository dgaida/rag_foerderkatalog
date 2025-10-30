#!/usr/bin/env python3
"""
main.py

Startpunkt für die RAG-Anwendung:
- Lädt CSV
- Führt (falls nötig) batch-basierte Embedding-Erzeugung durch (Ollama)
- Persistiert FAISS-Index
- Startet die Gradio-App (src.app)

Voraussetzungen:
- Ollama läuft lokal und das Embedding-Modell (z.B. 'nomic-embed-text') ist verfügbar.
- Module `src.search.engine`, `src.llm.llm_wrapper` und `src.utils.logging_config` sind vorhanden.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

from src.utils.logging_config import setup_logging, get_logger
from src.search.engine import ProjectSearchEngine
from src.llm.llm_wrapper import embed_text

logger = get_logger(__name__)


def batch_iter_rows(df, batch_size: int) -> Iterable[list]:
    """
    Liefert Zeilen-Indizes in Batches als Liste von (index, text)-Tupeln.
    """
    total = len(df)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = []
        for idx in range(start, end):
            row = df.iloc[idx]
            # Erzeuge ein kompaktes Dokument-Textfeld (Anpassbar)
            parts = []
            for col in ("Zuwendungsempfänger", "Zuwendungsempf\u00e4nger", "Thema", "Klartext Leistungsplansystematik"):
                if col in df.columns:
                    val = row.get(col, "")
                    if pd.notna(val):
                        parts.append(str(val))
            text = ". ".join(parts) if parts else str(row.to_dict())
            batch.append((idx, text))
        yield batch


def build_embeddings_in_batches(engine: ProjectSearchEngine, batch_size: int = 256, limit: int | None = None) -> None:
    """
    Erzeugt Embeddings in Batches für alle Dokumente in engine.df und fügt sie dem FAISS-Index hinzu.

    Args:
        engine: ProjectSearchEngine mit geladenem engine.df
        batch_size: Größe der Embedding-Batches (empfohlen: 64-1024 abhängig von RAM)
    """
    if engine.df is None:
        raise RuntimeError("DataFrame nicht geladen. Rufe load_and_clean() zuvor auf.")

    df = engine.df
    if limit is not None:
        logger.warning("⚠️ Debug-Modus: Indiziere nur die ersten %d Projekte.", limit)
        df = df.head(limit).copy()

    # Wenn Index schon Vektoren enthält -> überspringen
    if engine.faiss.index is not None and engine.faiss.index.ntotal > 0:
        logger.info("FAISS Index enthält bereits %d Vektoren — überspringe Batch-Erzeugung.", engine.faiss.index.ntotal)
        return

    logger.info("Starte Batch-Embedding: batch_size=%d, docs=%d", batch_size, len(engine.df))

    print("columns:", engine.df.columns)

    # Wir erzeugen Embeddings batchweise. Ollama's Python-API unterstützt in der Regel single-input oder Listen,
    # je nach Version; hier erzeugen wir sequenziell pro Dokument, aber in outer-Batches, damit man Fortschritt hat.
    added = 0
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(engine.df))
        batch_rows = []
        for idx in range(start, end):
            row = engine.df.iloc[idx]
            # Präferenz: Spalten für Text zusammensetzen (robust gegenüber verschiedenen Encodings)
            parts = []
            for col in ('="Zuwendungsempfänger"', "Zuwendungsempf\u00e4nger", '="Thema"',
                        '="Klartext Leistungsplansystematik"'):
                if col in engine.df.columns:
                    val = row.get(col)
                    if val is not None and str(val).strip() != "nan":
                        parts.append(str(val))
                else:
                    print(col, " ist nicht in ", engine.df.columns)
            text = ". ".join(parts) if parts else ""
            batch_rows.append((idx, text))

        # Embedding pro Item (seriell innerhalb Batch, um Ollama-API-Inkompatibilitäten zu vermeiden)
        for idx, text in batch_rows:
            try:
                vec = embed_text(text)
                if vec:  # Überprüfen Sie, ob der Vektor nicht leer ist
                    engine.faiss.add(vec, doc_id=str(idx), persist_now=False)
                    added += 1
                    if added % 100 == 0:
                        logger.info("Embeddings: hinzugefügt %d Dokumente...", added)
                else:
                    logger.warning("Leerer Embedding-Vektor für idx=%s", idx)
            except Exception as e:
                logger.exception("Fehler beim Erzeugen des Embeddings für idx=%s: %s", idx, e)

        # Nach jeder Batch einmal persistieren (sichert Fortschritt)
        engine.faiss.persist()
        logger.info("Batch %d-%d verarbeitet und persistiert.", start, end - 1)

    logger.info("Embeddings erzeugt und persistiert (insgesamt %d hinzugefügt).", added)


def parse_args():
    parser = argparse.ArgumentParser(description="Startet die RAG Förderprojekte App")
    parser.add_argument("--batch-size", "-b", type=int, default=256, help="Batch-Größe für Embedding-Erzeugung")
    parser.add_argument("--no-embeddings", action="store_true", help="Erzeuge keine Embeddings (falls Index vorhanden)")
    parser.add_argument("--log-level", default="INFO", help="Log-Level (DEBUG/INFO/WARNING/ERROR)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximale Anzahl an Projekten zum Indizieren (Debugging)")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    logger.info("Starte main.py")

    engine = ProjectSearchEngine()
    # CSV laden
    try:
        engine.load_and_clean()
    except Exception as e:
        logger.exception("Fehler beim Laden der CSV: %s", e)
        raise

    # Embeddings (batch)
    if not args.no_embeddings:
        try:
            build_embeddings_in_batches(engine, batch_size=args.batch_size, limit=args.limit)
        except Exception:
            logger.exception("Batch Embedding fehlgeschlagen")
            raise

    # Starte Gradio App (src.app)
    try:
        # Lokal importieren, damit vorherige Schritte (z.B. Logging) ausgeführt wurden
        from src.app import build_ui
        demo = build_ui(engine)
        demo.launch(share=False, inbrowser=True)
    except Exception as e:
        logger.exception("Fehler beim Starten der Gradio-App: %s", e)
        raise


if __name__ == "__main__":
    # pandas wird in batch_iter_rows referenziert; import hier, damit Datei top-level klein bleibt
    import pandas as pd  # type: ignore

    main()
