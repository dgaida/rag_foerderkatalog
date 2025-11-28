#!/usr/bin/env python3
"""
Index-Validierung und Synchronisation

Prüft, ob der FAISS-Index alle CSV-Einträge enthält und identifiziert
fehlende Projekte, die noch indiziert werden müssen.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

import pandas as pd

from ..embeddings.faiss_store import FaissStore
from .logging_config import get_logger

logger = get_logger(__name__)


class IndexValidator:
    """Validiert und synchronisiert FAISS-Index mit CSV-Daten.

    Diese Klasse prüft, ob alle Projekte aus der CSV-Datei im FAISS-Index
    vorhanden sind und identifiziert fehlende Einträge.

    Attributes:
        faiss: FaissStore-Instanz.
        df: Pandas DataFrame mit CSV-Daten.

    Example:
        >>> validator = IndexValidator(faiss_store, dataframe)
        >>> missing = validator.get_missing_indices()
        >>> if missing:
        ...     print(f"{len(missing)} Projekte fehlen im Index")
    """

    def __init__(self, faiss: FaissStore, df: pd.DataFrame):
        """Initialisiert den Validator.

        Args:
            faiss: FaissStore-Instanz mit geladenem Index.
            df: DataFrame mit CSV-Daten.
        """
        self.faiss = faiss
        self.df = df

    def get_indexed_ids(self) -> Set[int]:
        """Ermittelt alle IDs, die im FAISS-Index vorhanden sind.

        Returns:
            Set[int]: Menge aller indizierten Zeilen-IDs.

        Example:
            >>> validator = IndexValidator(faiss, df)
            >>> indexed = validator.get_indexed_ids()
            >>> len(indexed)
            150000
        """
        if self.faiss.id_map is None or len(self.faiss.id_map) == 0:
            return set()

        # id_map: {"0": "0", "1": "1", ...}
        # Die Values sind die ursprünglichen DataFrame-Indices als Strings
        indexed_ids = set()
        for doc_id in self.faiss.id_map.values():
            try:
                indexed_ids.add(int(doc_id))
            except (ValueError, TypeError):
                logger.warning("Ungültige doc_id im Index: %s", doc_id)
                continue

        return indexed_ids

    def get_csv_ids(self) -> Set[int]:
        """Ermittelt alle IDs aus dem CSV DataFrame.

        Returns:
            Set[int]: Menge aller CSV-Zeilen-IDs (DataFrame-Index).

        Example:
            >>> validator = IndexValidator(faiss, df)
            >>> csv_ids = validator.get_csv_ids()
            >>> len(csv_ids)
            200000
        """
        if self.df is None or self.df.empty:
            return set()

        return set(self.df.index)

    def get_missing_indices(self) -> List[int]:
        """Identifiziert fehlende Zeilen-IDs, die noch indiziert werden müssen.

        Returns:
            List[int]: Sortierte Liste von Zeilen-IDs, die nicht im Index sind.

        Example:
            >>> validator = IndexValidator(faiss, df)
            >>> missing = validator.get_missing_indices()
            >>> print(f"Fehlende Projekte: {len(missing)}")
            Fehlende Projekte: 50000
        """
        indexed_ids = self.get_indexed_ids()
        csv_ids = self.get_csv_ids()

        missing_ids = csv_ids - indexed_ids

        return sorted(missing_ids)

    def get_orphaned_indices(self) -> List[int]:
        """Identifiziert Index-Einträge ohne entsprechende CSV-Zeile.

        Dies kann passieren, wenn die CSV-Datei aktualisiert wurde und
        alte Projekte entfernt wurden.

        Returns:
            List[int]: Sortierte Liste von IDs im Index, die nicht in CSV sind.

        Example:
            >>> validator = IndexValidator(faiss, df)
            >>> orphaned = validator.get_orphaned_indices()
            >>> if orphaned:
            ...     print(f"Warnung: {len(orphaned)} verwaiste Einträge")
        """
        indexed_ids = self.get_indexed_ids()
        csv_ids = self.get_csv_ids()

        orphaned_ids = indexed_ids - csv_ids

        return sorted(orphaned_ids)

    def validate_index(self) -> Tuple[bool, dict]:
        """Führt vollständige Index-Validierung durch.

        Prüft:
        - Ob Index existiert
        - Ob alle CSV-Einträge indiziert sind
        - Ob verwaiste Einträge existieren
        - Synchronisationsstatus

        Returns:
            Tuple[bool, dict]: (is_valid, statistics)
                is_valid: True wenn Index vollständig synchron ist
                statistics: Dict mit detaillierten Statistiken

        Example:
            >>> validator = IndexValidator(faiss, df)
            >>> is_valid, stats = validator.validate_index()
            >>> print(f"Index vollständig: {is_valid}")
            >>> print(f"Fehlend: {stats['missing_count']}")
        """
        stats = {
            "csv_total": len(self.df) if self.df is not None else 0,
            "indexed_total": len(self.get_indexed_ids()),
            "missing_count": 0,
            "missing_ids": [],
            "orphaned_count": 0,
            "orphaned_ids": [],
            "is_empty": False,
            "sync_percentage": 0.0,
        }

        # Prüfe ob Index leer ist
        if self.faiss.index is None or self.faiss.index.ntotal == 0:
            stats["is_empty"] = True
            stats["missing_count"] = stats["csv_total"]
            stats["missing_ids"] = list(range(stats["csv_total"]))
            logger.warning("Index ist leer! Alle %d Projekte müssen indiziert werden.", stats["csv_total"])
            return False, stats

        # Ermittle fehlende und verwaiste Einträge
        missing = self.get_missing_indices()
        orphaned = self.get_orphaned_indices()

        stats["missing_count"] = len(missing)
        stats["missing_ids"] = missing[:100]  # Nur erste 100 für Log
        stats["orphaned_count"] = len(orphaned)
        stats["orphaned_ids"] = orphaned[:100]  # Nur erste 100 für Log

        # Berechne Synchronisations-Prozentsatz
        if stats["csv_total"] > 0:
            stats["sync_percentage"] = (stats["csv_total"] - stats["missing_count"]) / stats["csv_total"] * 100

        # Index ist valide wenn keine Einträge fehlen
        is_valid = stats["missing_count"] == 0

        return is_valid, stats

    def log_validation_report(self) -> None:
        """Erstellt und loggt einen detaillierten Validierungsbericht.

        Example:
            >>> validator = IndexValidator(faiss, df)
            >>> validator.log_validation_report()
        """
        is_valid, stats = self.validate_index()

        logger.info("═" * 60)
        logger.info("  Index-Validierungsbericht")
        logger.info("═" * 60)
        logger.info("")
        logger.info("📊 Statistiken:")
        logger.info("   CSV-Einträge gesamt:  %d", stats["csv_total"])
        logger.info("   Indizierte Einträge:  %d", stats["indexed_total"])
        logger.info("   Synchronisation:      %.1f%%", stats["sync_percentage"])
        logger.info("")

        if stats["is_empty"]:
            logger.warning("⚠️  Index ist LEER!")
            logger.warning("   Alle %d Projekte müssen indiziert werden.", stats["csv_total"])
        elif stats["missing_count"] > 0:
            logger.warning("⚠️  Fehlende Einträge:   %d", stats["missing_count"])
            if stats["missing_ids"]:
                logger.info("   Erste fehlende IDs: %s", stats["missing_ids"][:10])
        else:
            logger.info("✅ Index ist vollständig synchronisiert!")

        if stats["orphaned_count"] > 0:
            logger.warning("⚠️  Verwaiste Einträge:  %d", stats["orphaned_count"])
            logger.warning("   Diese IDs sind im Index, aber nicht in CSV:")
            if stats["orphaned_ids"]:
                logger.warning("   Erste verwaiste IDs: %s", stats["orphaned_ids"][:10])

        logger.info("")
        logger.info("═" * 60)

    def get_missing_projects(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Liefert DataFrame mit Projekten, die noch indiziert werden müssen.

        Args:
            limit: Optionales Limit für Anzahl der zurückgegebenen Projekte.

        Returns:
            pd.DataFrame: DataFrame mit fehlenden Projekten.

        Example:
            >>> validator = IndexValidator(faiss, df)
            >>> missing_df = validator.get_missing_projects(limit=1000)
            >>> print(f"Erste {len(missing_df)} fehlende Projekte:")
        """
        missing_ids = self.get_missing_indices()

        if not missing_ids:
            return pd.DataFrame()

        if limit:
            missing_ids = missing_ids[:limit]

        return self.df.loc[missing_ids].copy()


def check_index_completeness(
    faiss: FaissStore, df: pd.DataFrame, auto_index: bool = False, batch_size: int = 5000
) -> Tuple[bool, int]:
    """Convenience-Funktion zur Index-Prüfung beim App-Start.

    Prüft ob Index vollständig ist und gibt Empfehlung aus.
    Optional kann automatische Indizierung gestartet werden.

    Args:
        faiss: FaissStore-Instanz.
        df: DataFrame mit CSV-Daten.
        auto_index: Wenn True, werden fehlende Einträge automatisch indiziert.
        batch_size: Batch-Größe für automatische Indizierung.

    Returns:
        Tuple[bool, int]: (is_complete, missing_count)
            is_complete: True wenn Index vollständig ist
            missing_count: Anzahl fehlender Einträge

    Example:
        >>> from src.search.engine import ProjectSearchEngine
        >>> engine = ProjectSearchEngine()
        >>> engine.load_and_clean()
        >>> is_complete, missing = check_index_completeness(
        ...     engine.faiss, engine.df
        ... )
        >>> if not is_complete:
        ...     print(f"{missing} Projekte fehlen")
    """
    validator = IndexValidator(faiss, df)
    is_valid, stats = validator.validate_index()

    # Log-Report erstellen
    validator.log_validation_report()

    missing_count = stats["missing_count"]

    if not is_valid and missing_count > 0:
        logger.info("")
        logger.info("💡 Empfehlung:")

        if stats["is_empty"]:
            logger.info("   python main.py --batch-size %d", batch_size)
        else:
            logger.info("   Führen Sie erneut aus: python main.py --batch-size %d", batch_size)
            logger.info("   Die fehlenden %d Projekte werden dann indiziert.", missing_count)

        if auto_index:
            logger.info("")
            logger.info("🔄 Starte automatische Indizierung...")
            # Hier könnte die automatische Indizierung aufgerufen werden
            # Dies würde aber zu einer zirkulären Import-Abhängigkeit führen
            # Besser: Dies im main.py behandeln
            logger.warning("   auto_index=True wird derzeit nicht unterstützt.")
            logger.warning("   Bitte manuell starten.")

    return is_valid, missing_count


def get_new_projects_summary(validator: IndexValidator) -> str:
    """Erstellt eine lesbare Zusammenfassung neuer Projekte.

    Args:
        validator: IndexValidator-Instanz.

    Returns:
        str: Formatierte Zusammenfassung mit Beispielen.

    Example:
        >>> validator = IndexValidator(faiss, df)
        >>> summary = get_new_projects_summary(validator)
        >>> print(summary)
    """
    missing_df = validator.get_missing_projects(limit=5)

    if missing_df.empty:
        return "✅ Keine neuen Projekte gefunden."

    total_missing = len(validator.get_missing_indices())

    summary = f"📋 {total_missing} neue Projekte gefunden:\n\n"

    for idx, row in missing_df.iterrows():
        fkz = row.get('="FKZ"', "N/A")
        emp = row.get('="Zuwendungsempfänger"', "N/A")
        thema = row.get('="Thema"', "N/A")

        # Kürze lange Strings
        if len(emp) > 40:
            emp = emp[:37] + "..."
        if len(thema) > 60:
            thema = thema[:57] + "..."

        summary += f"  • FKZ {fkz}\n"
        summary += f"    {emp}\n"
        summary += f"    {thema}\n\n"

    if total_missing > 5:
        summary += f"  ... und {total_missing - 5} weitere Projekte\n"

    return summary
