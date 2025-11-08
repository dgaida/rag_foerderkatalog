#!/usr/bin/env python3
"""
main.py

Startpunkt für die RAG-Anwendung mit inkrementeller Indizierung und HuggingFace-Support.

Neue Features:
- Unterstützung für HuggingFace-Embeddings neben Ollama
- Provider-spezifische Indizes (vector.index vs. vector_hf.index)
- CLI-Parameter zur Auswahl des Embedding-Providers

Voraussetzungen:
- Für Ollama: Ollama läuft lokal mit Modell (z.B. 'nomic-embed-text')
- Für HuggingFace: llama-index-embeddings-huggingface installiert
"""
from __future__ import annotations

import argparse
import logging
import json

from src.utils.logging_config import setup_logging, get_logger
from src.search.engine import ProjectSearchEngine
from src.llm.llm_wrapper import embed_text
from src.config import get_index_files, HF_EMBED_MODEL_ALTERNATIVES, EmbeddingProvider
from src.utils.index_validator import IndexValidator, get_new_projects_summary

logger = get_logger(__name__)


def load_progress(provider: EmbeddingProvider) -> int:
    """Lädt den aktuellen Indizierungsfortschritt für einen Provider.

    Args:
        provider: "ollama" oder "huggingface"

    Returns:
        int: Anzahl der bereits indizierten Zeilen.
    """
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
    """Speichert den aktuellen Indizierungsfortschritt für einen Provider.

    Args:
        indexed_rows: Anzahl der bisher indizierten Zeilen.
        provider: "ollama" oder "huggingface"
    """
    try:
        _, _, progress_file = get_index_files(provider)
        progress_file.parent.mkdir(parents=True, exist_ok=True)

        data = {"indexed_rows": indexed_rows, "provider": provider}
        progress_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Fortschritt gespeichert (%s): %d Zeilen indiziert", provider, indexed_rows)
    except Exception as e:
        logger.error("Fehler beim Speichern des Fortschritts (%s): %s", provider, e)


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
        "📊 Indiziere %d von %d fehlenden Projekten mit %s (%.1f%%)",
        len(indices_to_process),
        total_missing,
        engine.provider,
        (len(indices_to_process) / total_missing) * 100,
    )

    added = 0
    for i, idx in enumerate(indices_to_process, 1):
        row = engine.df.iloc[idx]

        # Erstelle Embedding-Text
        text = engine._build_embedding_text(row)

        if not text:
            logger.warning("⚠️ Leerer Text für Zeile %d, überspringe", idx)
            continue

        try:
            vec = embed_text(text, provider=engine.provider, model=engine.embed_model)
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
        "✅ Batch abgeschlossen: %d Embeddings hinzugefügt (%s). " "Noch %d Projekte verbleibend (%.1f%%)",
        added,
        engine.provider,
        remaining,
        (remaining / total_missing) * 100 if total_missing > 0 else 0,
    )

    if remaining == 0:
        logger.info("🎉 FERTIG! Alle fehlenden Projekte wurden erfolgreich indiziert!")


def parse_args():
    parser = argparse.ArgumentParser(description="Startet die RAG Förderprojekte App mit Ollama oder HuggingFace Embeddings")

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
        help=("Embedding-Modell. Defaults:\n" "  Ollama: nomic-embed-text\n" "  HuggingFace: intfloat/e5-small-v2"),
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

    parser.add_argument("--show-missing", action="store_true", help="Zeige Details zu fehlenden Projekten")

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

    logger.info("🚀 Starte RAG Förderkatalog")
    logger.info("📦 Embedding-Provider: %s", args.provider.upper())
    if args.embed_model:
        logger.info("🎯 Embedding-Modell: %s", args.embed_model)

    # Zeige HuggingFace-Modelloptionen
    if args.provider == "huggingface" and not args.embed_model:
        logger.info("💡 Verfügbare HuggingFace-Modelle:")
        for model in HF_EMBED_MODEL_ALTERNATIVES:
            logger.info("   - %s", model)
        logger.info("")

    # Engine initialisieren
    engine = ProjectSearchEngine(provider=args.provider, embed_model=args.embed_model)

    # CSV laden
    try:
        engine.load_and_clean()
    except Exception as e:
        logger.exception("❌ Fehler beim Laden der CSV: %s", e)
        raise

    # Index-Validierung durchführen
    logger.info("")
    logger.info("🔍 Prüfe Index-Vollständigkeit (%s)...", args.provider)

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
                logger.warning("⚠️ Index wird neu aufgebaut (%s)!", args.provider)
                engine.faiss.clear()
                save_progress(0, args.provider)
                # Nach Clear ist der Index leer, re-validieren
                validator = IndexValidator(engine.faiss, engine.df)

            # Ermittle fehlende Einträge
            missing_indices = validator.get_missing_indices()

            if missing_indices:
                logger.info("")
                logger.info("🔄 Starte Indizierung der fehlenden Projekte (%s)...", args.provider)
                build_embeddings_for_missing(engine, missing_indices, batch_size=args.batch_size)

                # Re-validierung nach Indizierung
                logger.info("")
                logger.info("🔍 Erneute Validierung nach Indizierung (%s)...", args.provider)
                validator = IndexValidator(engine.faiss, engine.df)
                validator.log_validation_report()
            else:
                logger.info("✅ Index ist vollständig, keine Indizierung notwendig (%s)", args.provider)

        except Exception:
            logger.exception("❌ Indizierung fehlgeschlagen (%s)", args.provider)
            raise
    else:
        logger.info("⏭️ Indizierung übersprungen (--no-embeddings)")

        # Warnung wenn Index nicht vollständig
        if not is_complete and stats["missing_count"] > 0:
            logger.warning("")
            logger.warning("⚠️" * 30)
            logger.warning("⚠️  WARNUNG: Index ist nicht vollständig!")
            logger.warning("⚠️  %d Projekte fehlen im Index (%s)", stats["missing_count"], args.provider)
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
