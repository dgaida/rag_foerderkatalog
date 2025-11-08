#!/usr/bin/env python3
"""
main.py - AKTUALISIERT mit FKZ-basierter Validierung

Änderungen:
- Nutzt FKZIndexValidator statt IndexValidator
- Ermöglicht inkrementelle Updates bei geänderter CSV-Sortierung
- Erkennt neue Projekte unabhängig von ihrer Position in der CSV
"""
from __future__ import annotations

import argparse
import logging
import json

from src.utils.logging_config import setup_logging, get_logger
from src.search.engine import ProjectSearchEngine
from src.llm.llm_wrapper import embed_text
from src.config import get_index_files, EmbeddingProvider

# GEÄNDERT: Nutze FKZ-Validator
from src.utils.fkz_index_validator import FKZIndexValidator, get_projects_to_index

logger = get_logger(__name__)


def load_progress(provider: EmbeddingProvider) -> int:
    """Lädt den aktuellen Indizierungsfortschritt."""
    _, _, progress_file = get_index_files(provider)

    if progress_file.exists():
        try:
            data = json.loads(progress_file.read_text(encoding="utf-8"))
            return data.get("indexed_rows", 0)
        except Exception as e:
            logger.warning("Fehler beim Laden des Fortschritts (%s): %s", provider, e)
            return 0
    return 0


def save_progress(indexed_rows: int, provider: EmbeddingProvider) -> None:
    """Speichert den aktuellen Indizierungsfortschritt."""
    try:
        _, _, progress_file = get_index_files(provider)
        progress_file.parent.mkdir(parents=True, exist_ok=True)

        data = {"indexed_rows": indexed_rows, "provider": provider}
        progress_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Fortschritt gespeichert (%s): %d Zeilen indiziert", provider, indexed_rows)
    except Exception as e:
        logger.error("Fehler beim Speichern des Fortschritts (%s): %s", provider, e)


def index_new_projects_fkz(engine: ProjectSearchEngine, new_projects_df, batch_size: int = 5000) -> None:
    """Indiziert neue Projekte basierend auf FKZ-Validierung.

    Args:
        engine: ProjectSearchEngine mit geladenem DataFrame
        new_projects_df: DataFrame mit neuen Projekten
        batch_size: Maximale Anzahl pro Durchlauf
    """
    if new_projects_df.empty:
        logger.info("✅ Keine neuen Projekte zum Indizieren")
        return

    total_new = len(new_projects_df)
    batch_end = min(batch_size, total_new)
    projects_to_process = new_projects_df.head(batch_end)

    logger.info(
        "📊 Indiziere %d von %d neuen Projekten mit %s (%.1f%%)",
        len(projects_to_process),
        total_new,
        engine.provider,
        (len(projects_to_process) / total_new) * 100,
    )

    added = 0
    for i, (idx, row) in enumerate(projects_to_process.iterrows(), 1):
        # Erstelle Embedding-Text
        text = engine._build_embedding_text(row)

        if not text:
            fkz = row.get('="FKZ"', "UNBEKANNT")
            logger.warning("⚠️ Leerer Text für FKZ %s, überspringe", fkz)
            continue

        try:
            vec = embed_text(text, provider=engine.provider, model=engine.embed_model)
            if vec:
                # WICHTIG: Speichere DataFrame-Index als doc_id
                engine.faiss.add(vec, doc_id=str(idx), persist_now=False)
                added += 1

                # Fortschritt alle 100 Zeilen loggen
                if added % 100 == 0:
                    logger.info(
                        "⏳ Fortschritt: %d/%d (%.1f%%)",
                        added,
                        len(projects_to_process),
                        (added / len(projects_to_process)) * 100,
                    )
            else:
                fkz = row.get('="FKZ"', "UNBEKANNT")
                logger.warning("⚠️ Leerer Embedding-Vektor für FKZ=%s", fkz)
        except Exception as e:
            fkz = row.get('="FKZ"', "UNBEKANNT")
            logger.exception("❌ Fehler beim Erzeugen des Embeddings für FKZ=%s: %s", fkz, e)

    # Persistiere Index
    engine.faiss.persist()

    remaining = total_new - len(projects_to_process)
    logger.info(
        "✅ Batch abgeschlossen: %d Embeddings hinzugefügt (%s). " "Noch %d Projekte verbleibend (%.1f%%)",
        added,
        engine.provider,
        remaining,
        (remaining / total_new) * 100 if total_new > 0 else 0,
    )

    if remaining == 0:
        logger.info("🎉 FERTIG! Alle neuen Projekte wurden erfolgreich indiziert!")


def parse_args():
    parser = argparse.ArgumentParser(description="Startet die RAG Förderprojekte App mit FKZ-basierter Validierung")

    # Provider Selection
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        choices=["ollama", "huggingface"],
        default="ollama",
        help="Embedding-Provider (ollama oder huggingface, Standard: ollama)",
    )

    parser.add_argument(
        "--embed-model",
        "-m",
        type=str,
        default=None,
        help="Embedding-Modell",
    )

    # Indexing Options
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=5000,
        help="Anzahl der Zeilen pro Indizierungs-Batch (Standard: 5000)",
    )

    parser.add_argument("--no-embeddings", action="store_true", help="Überspringe Indizierung (nur App starten)")

    parser.add_argument("--force-rebuild", action="store_true", help="Index neu aufbauen (von vorne beginnen)")

    # Validation Options
    parser.add_argument("--validate-only", action="store_true", help="Nur Index validieren, nicht starten")

    parser.add_argument("--show-missing", action="store_true", help="Zeige Details zu neuen Projekten")

    parser.add_argument("--index-info", action="store_true", help="Zeige Informationen über alle Indizes")

    # Logging
    parser.add_argument("--log-level", default="INFO", help="Log-Level (DEBUG/INFO/WARNING/ERROR)")

    return parser.parse_args()


def show_all_index_info():
    """Zeigt Informationen über alle verfügbaren Indizes."""
    logger.info("═" * 60)
    logger.info("  Index-Übersicht")
    logger.info("═" * 60)
    logger.info("")

    for provider in ["ollama", "huggingface"]:
        logger.info("🔍 Provider: %s", provider.upper())
        try:
            engine = ProjectSearchEngine(provider=provider)
            engine.load_and_clean()
            info = engine.get_index_info()

            logger.info("   Index-Datei: %s", info["index_file"])
            logger.info("   Existiert: %s", "✅" if info["exists"] else "❌")
            logger.info("   Dimension: %s", info["dimension"])
            logger.info("   Vektoren: %d", info["total_vectors"])
            logger.info("   CSV-Zeilen: %d", info["csv_rows"])

            if info["total_vectors"] > 0 and info["csv_rows"] > 0:
                coverage = (info["total_vectors"] / info["csv_rows"]) * 100
                logger.info("   Abdeckung: %.1f%%", coverage)

            logger.info("")
        except Exception as e:
            logger.error("   Fehler beim Laden: %s", e)
            logger.info("")

    logger.info("═" * 60)


def main():
    args = parse_args()
    setup_logging()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    # Index-Info anzeigen und beenden
    if args.index_info:
        show_all_index_info()
        return

    logger.info("🚀 Starte RAG Förderkatalog (FKZ-basierte Validierung)")
    logger.info("📦 Embedding-Provider: %s", args.provider.upper())
    if args.embed_model:
        logger.info("🎯 Embedding-Modell: %s", args.embed_model)

    # Engine initialisieren
    engine = ProjectSearchEngine(provider=args.provider, embed_model=args.embed_model)

    # CSV laden
    try:
        engine.load_and_clean()
    except Exception as e:
        logger.exception("❌ Fehler beim Laden der CSV: %s", e)
        raise

    # ===== GEÄNDERT: FKZ-basierte Validierung =====
    logger.info("")
    logger.info("🔍 Prüfe Index-Vollständigkeit (FKZ-basiert)...")

    validator = FKZIndexValidator(engine.faiss, engine.df)
    is_complete, stats = validator.validate()

    # Detaillierter Report
    validator.log_validation_report()

    # Zeige neue Projekte wenn gewünscht
    if args.show_missing and stats["new_count"] > 0:
        logger.info("")
        summary = validator.get_new_projects_summary(limit=10)
        logger.info(summary)

    # Nur Validierung, dann beenden
    if args.validate_only:
        logger.info("✅ Validierung abgeschlossen (--validate-only)")
        return

    # Indizierung
    if not args.no_embeddings:
        try:
            if args.force_rebuild:
                logger.warning("⚠️ Index wird neu aufgebaut (%s)!", args.provider)
                engine.faiss.clear()
                save_progress(0, args.provider)
                # Nach Clear ist der Index leer, re-validieren
                validator = FKZIndexValidator(engine.faiss, engine.df)

            # Ermittle neue Projekte (FKZ-basiert)
            new_projects_df = get_projects_to_index(engine.faiss, engine.df, batch_size=args.batch_size)

            if not new_projects_df.empty:
                logger.info("")
                logger.info("🔄 Starte Indizierung der neuen Projekte (%s)...", args.provider)
                index_new_projects_fkz(engine, new_projects_df, batch_size=args.batch_size)

                # Re-validierung nach Indizierung
                logger.info("")
                logger.info("🔍 Erneute Validierung nach Indizierung (%s)...", args.provider)
                validator = FKZIndexValidator(engine.faiss, engine.df)
                validator.log_validation_report()
            else:
                logger.info("✅ Index ist vollständig, keine Indizierung notwendig (%s)", args.provider)

        except Exception:
            logger.exception("❌ Indizierung fehlgeschlagen (%s)", args.provider)
            raise
    else:
        logger.info("⏭️ Indizierung übersprungen (--no-embeddings)")

        # Warnung wenn Index nicht vollständig
        if not is_complete and stats["new_count"] > 0:
            logger.warning("")
            logger.warning("⚠️" * 30)
            logger.warning("⚠️  WARNUNG: Index ist nicht vollständig!")
            logger.warning("⚠️  %d Projekte fehlen im Index (%s)", stats["new_count"], args.provider)
            logger.warning("⚠️  Suchergebnisse könnten unvollständig sein.")
            logger.warning("⚠️")
            logger.warning("⚠️  Zum Vervollständigen starten Sie:")
            logger.warning("⚠️    python main.py --provider %s --batch-size %d", args.provider, args.batch_size)
            logger.warning("⚠️" * 30)
            logger.warning("")

    # Starte Gradio App
    try:
        from src.app import build_ui

        demo = build_ui(engine)

        logger.info("")
        logger.info("🌐 Starte Gradio-Oberfläche...")
        logger.info("📊 Provider: %s", args.provider.upper())
        logger.info("📁 Index: %s", engine.faiss.index_file.name)
        demo.launch(share=False, inbrowser=True)
    except Exception as e:
        logger.exception("❌ Fehler beim Starten der Gradio-App: %s", e)
        raise


if __name__ == "__main__":
    main()
