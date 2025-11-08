#!/usr/bin/env python3
"""
Beispiel-Script zum Vergleich von Ollama vs HuggingFace Embeddings

Dieses Script demonstriert:
1. Verwendung beider Provider
2. Vergleich der Suchergebnisse
3. Performance-Messung
"""

import time

# from pathlib import Path
# import sys

# Füge src zum Python-Path hinzu
# sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search.engine import ProjectSearchEngine
from src.utils.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def compare_search_results(query: str, k: int = 10):
    """Vergleicht Suchergebnisse zwischen Ollama und HuggingFace.

    Args:
        query: Suchanfrage
        k: Anzahl Treffer
    """
    logger.info("=" * 80)
    logger.info("Vergleiche Suchergebnisse für: '%s'", query)
    logger.info("=" * 80)
    logger.info("")

    results = {}

    for provider in ["ollama", "huggingface"]:
        logger.info("🔍 Provider: %s", provider.upper())
        logger.info("-" * 80)

        try:
            # Engine initialisieren
            engine = ProjectSearchEngine(provider=provider)
            engine.load_and_clean()

            # Index-Info
            info = engine.get_index_info()
            logger.info("   Index: %s", info["index_file"])
            logger.info("   Vektoren: %d", info["total_vectors"])
            logger.info("   Dimension: %d", info["dimension"])

            if info["total_vectors"] == 0:
                logger.warning("   ⚠️ Index ist leer! Bitte zuerst indizieren.")
                logger.info("")
                continue

            # Suche mit Zeitmessung
            start_time = time.time()
            df_results = engine.search(query, k=k)
            search_time = time.time() - start_time

            logger.info("   ⏱️ Suchzeit: %.3f Sekunden", search_time)
            logger.info("   📊 Treffer: %d", len(df_results))

            if not df_results.empty:
                # Top-3 Ergebnisse anzeigen
                logger.info("")
                logger.info("   Top-3 Ergebnisse:")
                for i, (idx, row) in enumerate(df_results.head(3).iterrows(), 1):
                    fkz = row.get('="FKZ"', "N/A")
                    score = row.get("__score", 0.0)
                    thema = row.get('="Thema"', "N/A")

                    # Kürze Thema wenn zu lang
                    if len(thema) > 60:
                        thema = thema[:57] + "..."

                    logger.info("   %d. [%.4f] %s", i, score, fkz)
                    logger.info("      %s", thema)

                # Speichere für Vergleich
                results[provider] = {
                    "df": df_results,
                    "time": search_time,
                    "dimension": info["dimension"],
                }

            logger.info("")

        except Exception as e:
            logger.error("   ❌ Fehler: %s", e)
            logger.info("")
            continue

    # Vergleich
    if len(results) == 2:
        logger.info("=" * 80)
        logger.info("📈 VERGLEICH")
        logger.info("=" * 80)

        ollama_results = results.get("ollama")
        hf_results = results.get("huggingface")

        if ollama_results and hf_results:
            logger.info("")
            logger.info("Geschwindigkeit:")
            logger.info("   Ollama:      %.3f Sekunden", ollama_results["time"])
            logger.info("   HuggingFace: %.3f Sekunden", hf_results["time"])

            speedup = ollama_results["time"] / hf_results["time"]
            if speedup > 1:
                logger.info("   → HuggingFace ist %.1fx schneller", speedup)
            else:
                logger.info("   → Ollama ist %.1fx schneller", 1 / speedup)

            logger.info("")
            logger.info("Embedding-Dimension:")
            logger.info("   Ollama:      %d", ollama_results["dimension"])
            logger.info("   HuggingFace: %d", hf_results["dimension"])

            # Überlappung der Top-10
            ollama_fkz = set(ollama_results["df"].head(10)['="FKZ"'].values)
            hf_fkz = set(hf_results["df"].head(10)['="FKZ"'].values)
            overlap = len(ollama_fkz.intersection(hf_fkz))

            logger.info("")
            logger.info("Übereinstimmung (Top-10):")
            logger.info("   Gemeinsame Treffer: %d / 10 (%.0f%%)", overlap, overlap * 10)

            if overlap >= 8:
                logger.info("   ✅ Sehr hohe Übereinstimmung")
            elif overlap >= 5:
                logger.info("   ⚠️ Moderate Übereinstimmung")
            else:
                logger.info("   ❌ Geringe Übereinstimmung")

    logger.info("")
    logger.info("=" * 80)


def main():
    """Hauptfunktion mit verschiedenen Test-Queries."""

    test_queries = [
        "Künstliche Intelligenz Hochschule Bayern",
        "Wasserstoff Energie NRW",
        "Quantencomputing Forschung",
        "Klimawandel Digitalisierung",
    ]

    logger.info("")
    logger.info("🔬 Provider-Vergleich: Ollama vs HuggingFace")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Dieses Script vergleicht Suchergebnisse zwischen:")
    logger.info("  • Ollama (nomic-embed-text, 768 dim)")
    logger.info("  • HuggingFace (intfloat/e5-small-v2, 384 dim)")
    logger.info("")
    logger.info("Hinweis: Beide Indizes müssen existieren!")
    logger.info("         Falls nicht: python main.py --provider [ollama|huggingface]")
    logger.info("")

    input("Drücken Sie Enter zum Starten...")
    logger.info("")

    for query in test_queries:
        compare_search_results(query, k=10)
        logger.info("")

        # Warte auf Nutzer-Input für nächste Query
        if query != test_queries[-1]:
            input("Drücken Sie Enter für nächste Query...")
            logger.info("")

    logger.info("✅ Vergleich abgeschlossen!")


if __name__ == "__main__":
    main()
