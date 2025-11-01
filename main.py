#!/usr/bin/env python3
"""
main.py

Startpunkt für die RAG-Anwendung mit inkrementeller Indizierung:
- Lädt CSV
- Validiert Index-Vollständigkeit
- Indiziert fehlende Projekte automatisch
- Startet die Gradio-App

Voraussetzungen:
- Ollama läuft lokal und das Embedding-Modell (z.B. 'nomic-embed-text') ist verfügbar.
"""
from __future__ import annotations

import argparse
import logging
import json

from src.utils.logging_config import setup_logging, get_logger
from src.search.engine import ProjectSearchEngine
from src.llm.llm_wrapper import embed_text
from src.config import DATA_DIR
from src.utils.index_validator import IndexValidator, get_new_projects_summary

logger = get_logger(__name__)

# Datei zum Speichern des Fortschritts
PROGRESS_FILE = DATA_DIR / "indexing_progress.json"


def load_progress() -> int:
    """Lädt den aktuellen Indizierungsfortschritt.

    Returns:
        int: Anzahl der bereits indizierten Zeilen.
    """
    if PROGRESS_FILE.exists():
        try:
            data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
            return data.get("indexed_rows", 0)
        except Exception as e:
            logger.warning("Fehler beim Laden des Fortschritts: %s", e)
            return 0
    return 0


def save_progress(indexed_rows: int) -> None:
    """Speichert den aktuellen Indizierungsfortschritt.

    Args:
        indexed_rows: Anzahl der bisher indizierten Zeilen.
    """
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {"indexed_rows": indexed_rows}
        PROGRESS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Fortschritt gespeichert: %d Zeilen indiziert", indexed_rows)
    except Exception as e:
        logger.error("Fehler beim Speichern des Fortschritts: %s", e)


def build_embeddings_for_missing(engine: ProjectSearchEngine, missing_indices: list, batch_size: int = 5000) -> None:
    """Indiziert nur die fehlenden Projekte.

    Args:
        engine: ProjectSearchEngine mit geladenem engine.df
        missing_indices: Liste der fehlenden DataFrame-Indizes
        batch_size: Maximale Anzahl pro Durchlauf
    """
    if not missing_indices:
        logger.info("✅ Keine fehlenden Projekte zum Indizieren")
        return

    total_missing = len(missing_indices)
    batch_end = min(batch_size, total_missing)
    indices_to_process = missing_indices[:batch_end]

    logger.info(
        "📊 Indiziere %d von %d fehlenden Projekten (%.1f%%)",
        len(indices_to_process),
        total_missing,
        (len(indices_to_process) / total_missing) * 100,
    )

    added = 0
    for i, idx in enumerate(indices_to_process, 1):
        row = engine.df.iloc[idx]

        # Erstelle Embedding-Text
        parts = []
        for col in [
            '="Zuwendungsempfänger"',
            '="Thema"',
            '="Klartext Leistungsplansystematik"',
            '="Ausführende Stelle"',
            '="Stadt/Gemeinde"',
            '="Bundesland"',
            "__laufzeit",
            '="Förderprofil"',
            '="Verbundprojekt"',
        ]:
            if col in engine.df.columns:
                val = row.get(col)
                if val is not None and str(val).strip() and str(val).strip() != "nan":
                    parts.append(str(val))

        text = ". ".join(parts) if parts else ""

        if not text:
            logger.warning("⚠️ Leerer Text für Zeile %d, überspringe", idx)
            continue

        try:
            vec = embed_text(text)
            if vec:
                engine.faiss.add(vec, doc_id=str(idx), persist_now=False)
                added += 1

                # Fortschritt alle 100 Zeilen loggen
                if added % 100 == 0:
                    logger.info(
                        "⏳ Fortschritt: %d/%d (%.1f%%)",
                        added,
                        len(indices_to_process),
                        (added / len(indices_to_process)) * 100,
                    )
            else:
                logger.warning("⚠️ Leerer Embedding-Vektor für idx=%d", idx)
        except Exception as e:
            logger.exception("❌ Fehler beim Erzeugen des Embeddings für idx=%d: %s", idx, e)

    # Persistiere Index
    engine.faiss.persist()

    remaining = total_missing - len(indices_to_process)
    logger.info(
        "✅ Batch abgeschlossen: %d Embeddings hinzugefügt. " "Noch %d Projekte verbleibend (%.1f%%)",
        added,
        remaining,
        (remaining / total_missing) * 100 if total_missing > 0 else 0,
    )

    if remaining == 0:
        logger.info("🎉 FERTIG! Alle fehlenden Projekte wurden erfolgreich indiziert!")


def build_embeddings_incremental(engine: ProjectSearchEngine, batch_size: int = 5000, force_rebuild: bool = False) -> None:
    """Indiziert inkrementell weitere Projekte (Legacy-Funktion).

    DEPRECATED: Verwenden Sie stattdessen die Validator-basierte Methode.

    Bei jedem Aufruf werden batch_size weitere Zeilen indiziert,
    bis alle Projekte verarbeitet sind.

    Args:
        engine: ProjectSearchEngine mit geladenem engine.df
        batch_size: Anzahl der Zeilen, die pro Start indiziert werden
        force_rebuild: Wenn True, wird von vorne begonnen
    """
    logger.warning("⚠️ Verwende Legacy-Indizierung. Empfohlen: Validator-basierte Methode")

    if engine.df is None:
        raise RuntimeError("DataFrame nicht geladen. Rufe load_and_clean() zuvor auf.")

    total_rows = len(engine.df)

    # Lade Fortschritt
    if force_rebuild:
        start_idx = 0
        logger.info("🔄 Neustart der Indizierung erzwungen")
    else:
        start_idx = load_progress()

    if start_idx >= total_rows:
        logger.info("✅ Alle %d Projekte sind bereits indiziert!", total_rows)
        return

    end_idx = min(start_idx + batch_size, total_rows)

    logger.info(
        "📊 Indiziere Zeilen %d bis %d von %d (%.1f%% abgeschlossen)",
        start_idx,
        end_idx,
        total_rows,
        (start_idx / total_rows) * 100,
    )

    # Indiziere den aktuellen Batch
    added = 0
    for idx in range(start_idx, end_idx):
        row = engine.df.iloc[idx]

        # Erstelle Embedding-Text
        parts = []
        for col in [
            '="Zuwendungsempfänger"',
            '="Thema"',
            '="Klartext Leistungsplansystematik"',
            '="Ausführende Stelle"',
            '="Stadt/Gemeinde"',
            '="Bundesland"',
            "__laufzeit",
            '="Förderprofil"',
            '="Verbundprojekt"',
        ]:
            if col in engine.df.columns:
                val = row.get(col)
                if val is not None and str(val).strip() and str(val).strip() != "nan":
                    parts.append(str(val))

        text = ". ".join(parts) if parts else ""

        if not text:
            logger.warning("⚠️ Leerer Text für Zeile %d, überspringe", idx)
            continue

        try:
            vec = embed_text(text)
            if vec:
                engine.faiss.add(vec, doc_id=str(idx), persist_now=False)
                added += 1

                # Fortschritt alle 100 Zeilen loggen
                if added % 100 == 0:
                    current = start_idx + added
                    logger.info("⏳ Fortschritt: %d/%d (%.1f%%)", current, total_rows, (current / total_rows) * 100)
            else:
                logger.warning("⚠️ Leerer Embedding-Vektor für idx=%d", idx)
        except Exception as e:
            logger.exception("❌ Fehler beim Erzeugen des Embeddings für idx=%d: %s", idx, e)

    # Persistiere Index und Fortschritt
    engine.faiss.persist()
    save_progress(end_idx)

    remaining = total_rows - end_idx
    logger.info(
        "✅ Batch abgeschlossen: %d Embeddings hinzugefügt. " "Noch %d Projekte verbleibend (%.1f%%)",
        added,
        remaining,
        (remaining / total_rows) * 100,
    )

    if end_idx >= total_rows:
        logger.info("🎉 FERTIG! Alle %d Projekte wurden erfolgreich indiziert!", total_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Startet die RAG Förderprojekte App")
    parser.add_argument(
        "--batch-size", "-b", type=int, default=5000, help="Anzahl der Zeilen, die pro Start indiziert werden (Standard: 5000)"
    )
    parser.add_argument("--no-embeddings", action="store_true", help="Überspringe Indizierung (nur App starten)")
    parser.add_argument("--force-rebuild", action="store_true", help="Index neu aufbauen (von vorne beginnen)")
    parser.add_argument("--validate-only", action="store_true", help="Nur Index validieren, nicht starten")
    parser.add_argument("--show-missing", action="store_true", help="Zeige Details zu fehlenden Projekten")
    parser.add_argument("--log-level", default="INFO", help="Log-Level (DEBUG/INFO/WARNING/ERROR)")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    logger.info("🚀 Starte RAG Förderkatalog mit intelligenter Index-Validierung")

    engine = ProjectSearchEngine()

    # CSV laden
    try:
        engine.load_and_clean()
    except Exception as e:
        logger.exception("❌ Fehler beim Laden der CSV: %s", e)
        raise

    # Index-Validierung durchführen
    logger.info("")
    logger.info("🔍 Prüfe Index-Vollständigkeit...")

    validator = IndexValidator(engine.faiss, engine.df)
    is_complete, stats = validator.validate_index()

    # Detaillierter Report
    validator.log_validation_report()

    # Zeige neue Projekte wenn gewünscht
    if args.show_missing and stats["missing_count"] > 0:
        logger.info("")
        summary = get_new_projects_summary(validator)
        logger.info(summary)

    # Nur Validierung, dann beenden
    if args.validate_only:
        logger.info("✅ Validierung abgeschlossen (--validate-only)")
        return

    # Indizierung
    if not args.no_embeddings:
        try:
            if args.force_rebuild:
                logger.warning("⚠️ Index wird neu aufgebaut!")
                engine.faiss.clear()
                save_progress(0)
                # Nach Clear ist der Index leer, re-validieren
                validator = IndexValidator(engine.faiss, engine.df)

            # Ermittle fehlende Einträge
            missing_indices = validator.get_missing_indices()

            if missing_indices:
                logger.info("")
                logger.info("🔄 Starte Indizierung der fehlenden Projekte...")
                build_embeddings_for_missing(engine, missing_indices, batch_size=args.batch_size)

                # Re-validierung nach Indizierung
                logger.info("")
                logger.info("🔍 Erneute Validierung nach Indizierung...")
                validator = IndexValidator(engine.faiss, engine.df)
                validator.log_validation_report()
            else:
                logger.info("✅ Index ist vollständig, keine Indizierung notwendig")

        except Exception:
            logger.exception("❌ Indizierung fehlgeschlagen")
            raise
    else:
        logger.info("⏭️ Indizierung übersprungen (--no-embeddings)")

        # Warnung wenn Index nicht vollständig
        if not is_complete and stats["missing_count"] > 0:
            logger.warning("")
            logger.warning("⚠️" * 30)
            logger.warning("⚠️  WARNUNG: Index ist nicht vollständig!")
            logger.warning("⚠️  %d Projekte fehlen im Index", stats["missing_count"])
            logger.warning("⚠️  Suchergebnisse könnten unvollständig sein.")
            logger.warning("⚠️")
            logger.warning("⚠️  Zum Vervollständigen starten Sie:")
            logger.warning("⚠️    python main.py --batch-size %d", args.batch_size)
            logger.warning("⚠️" * 30)
            logger.warning("")

    # Starte Gradio App
    try:
        from src.app import build_ui

        demo = build_ui(engine)

        logger.info("")
        logger.info("🌐 Starte Gradio-Oberfläche...")
        demo.launch(share=False, inbrowser=True)
    except Exception as e:
        logger.exception("❌ Fehler beim Starten der Gradio-App: %s", e)
        raise


if __name__ == "__main__":
    main()
