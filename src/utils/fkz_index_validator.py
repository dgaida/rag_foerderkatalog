#!/usr/bin/env python3
"""
FKZ-basierte Index-Validierung für robuste Projekt-Synchronisation

Diese erweiterte Validierung nutzt FKZ statt DataFrame-Indizes,
um neue Projekte auch bei geänderter CSV-Sortierung zu erkennen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Tuple

import pandas as pd

from src.embeddings.faiss_store import FaissStore
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class FKZIndexValidator:
    """FKZ-basierte Index-Validierung.

    Vorteile gegenüber der Index-basierten Validierung:
    - Robust gegen Umsortierung der CSV
    - Erkennt neue Projekte unabhängig von ihrer Position
    - Ermöglicht inkrementelle Updates ohne Re-Indexierung

    Attributes:
        faiss: FaissStore-Instanz mit Index
        df: DataFrame mit CSV-Daten (muss '="FKZ"' Spalte haben)
        fkz_column: Name der FKZ-Spalte (default: '="FKZ"')

    Example:
        >>> validator = FKZIndexValidator(faiss_store, dataframe)
        >>> new_projects = validator.get_new_projects()
        >>> print(f"{len(new_projects)} neue Projekte gefunden")
    """

    def __init__(self, faiss: FaissStore, df: pd.DataFrame, fkz_column: str = '="FKZ"'):
        """Initialisiert den FKZ-Validator.

        Args:
            faiss: FaissStore mit geladenem Index
            df: DataFrame mit Projektdaten
            fkz_column: Name der FKZ-Spalte

        Raises:
            ValueError: Wenn FKZ-Spalte nicht existiert
        """
        self.faiss = faiss
        self.df = df
        self.fkz_column = fkz_column

        if fkz_column not in df.columns:
            raise ValueError(
                f"FKZ-Spalte '{fkz_column}' nicht in DataFrame gefunden. " f"Verfügbare Spalten: {', '.join(df.columns)}"
            )

    def get_indexed_fkz(self) -> Set[str]:
        """Ermittelt alle FKZ, die im Index vorhanden sind.

        Der Index speichert DataFrame-Row-Indizes als doc_id.
        Wir müssen diese zurück zu FKZ mappen.

        Returns:
            Set[str]: Menge aller indizierten FKZ

        Example:
            >>> indexed = validator.get_indexed_fkz()
            >>> '13BDB60030' in indexed
            True
        """
        if not self.faiss.id_map or len(self.faiss.id_map) == 0:
            return set()

        indexed_fkz = set()

        # id_map: {"0": "0", "1": "1", ...}
        # Die Values sind DataFrame-Indizes
        for faiss_id, df_idx_str in self.faiss.id_map.items():
            try:
                df_idx = int(df_idx_str)

                # Hole FKZ für diesen Index
                if df_idx < len(self.df):
                    fkz = self.df.iloc[df_idx][self.fkz_column]

                    # Bereinige FKZ (entferne mögliche Formatierung)
                    if pd.notna(fkz):
                        fkz_clean = str(fkz).strip()
                        if fkz_clean and fkz_clean not in ("nan", "<NA>"):
                            indexed_fkz.add(fkz_clean)

            except (ValueError, TypeError, KeyError, IndexError) as e:
                logger.warning("Fehler beim Mapping von Index %s: %s", df_idx_str, e)
                continue

        return indexed_fkz

    def get_csv_fkz(self) -> Set[str]:
        """Ermittelt alle FKZ aus der CSV.

        Returns:
            Set[str]: Menge aller FKZ in der CSV

        Example:
            >>> csv_fkz = validator.get_csv_fkz()
            >>> len(csv_fkz)
            234567
        """
        if self.df is None or self.df.empty:
            return set()

        # Extrahiere und bereinige FKZ
        fkz_series = self.df[self.fkz_column].astype(str).str.strip()

        # Filtere leere, 'nan' und '<NA>' Werte
        fkz_set = set(fkz for fkz in fkz_series if pd.notna(fkz) and fkz not in ("", "nan", "<NA>"))

        return fkz_set

    def get_new_fkz(self) -> Set[str]:
        """Identifiziert FKZ, die in CSV aber nicht im Index sind.

        Dies sind die neuen Projekte, die indiziert werden müssen.

        Returns:
            Set[str]: Menge aller neuen FKZ

        Example:
            >>> new = validator.get_new_fkz()
            >>> print(f"{len(new)} neue Projekte")
        """
        indexed = self.get_indexed_fkz()
        csv = self.get_csv_fkz()

        new_fkz = csv - indexed
        return new_fkz

    def get_removed_fkz(self) -> Set[str]:
        """Identifiziert FKZ, die im Index aber nicht mehr in CSV sind.

        Diese Projekte wurden aus der CSV entfernt (veraltet).

        Returns:
            Set[str]: Menge aller entfernten FKZ
        """
        indexed = self.get_indexed_fkz()
        csv = self.get_csv_fkz()

        removed_fkz = indexed - csv
        return removed_fkz

    def get_new_projects(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Liefert DataFrame mit neuen Projekten.

        Args:
            limit: Optionales Limit für Anzahl

        Returns:
            pd.DataFrame: Neue Projekte

        Example:
            >>> new_df = validator.get_new_projects(limit=1000)
            >>> print(f"Erste {len(new_df)} neue Projekte")
        """
        new_fkz = self.get_new_fkz()

        if not new_fkz:
            return pd.DataFrame()

        # Filtere DataFrame
        mask = self.df[self.fkz_column].astype(str).str.strip().isin(new_fkz)
        new_df = self.df[mask].copy()

        if limit and len(new_df) > limit:
            new_df = new_df.head(limit)

        return new_df

    def validate(self) -> Tuple[bool, dict]:
        """Führt vollständige FKZ-basierte Validierung durch.

        Returns:
            Tuple[bool, dict]: (is_complete, statistics)
                is_complete: True wenn alle CSV-Projekte indiziert
                statistics: Detaillierte Statistiken

        Example:
            >>> is_valid, stats = validator.validate()
            >>> print(f"Neu: {stats['new_count']}")
            >>> print(f"Entfernt: {stats['removed_count']}")
        """
        indexed_fkz = self.get_indexed_fkz()
        csv_fkz = self.get_csv_fkz()
        new_fkz = self.get_new_fkz()
        removed_fkz = self.get_removed_fkz()

        stats = {
            "csv_total": len(csv_fkz),
            "indexed_total": len(indexed_fkz),
            "new_count": len(new_fkz),
            "removed_count": len(removed_fkz),
            "new_fkz": sorted(list(new_fkz))[:100],  # Erste 100
            "removed_fkz": sorted(list(removed_fkz))[:100],
            "is_empty": len(indexed_fkz) == 0,
            "sync_percentage": 0.0,
        }

        # Berechne Synchronisation
        if stats["csv_total"] > 0:
            synced = len(indexed_fkz.intersection(csv_fkz))
            stats["sync_percentage"] = (synced / stats["csv_total"]) * 100

        is_complete = stats["new_count"] == 0

        return is_complete, stats

    def log_validation_report(self) -> None:
        """Erstellt detaillierten Validierungsbericht.

        Example:
            >>> validator.log_validation_report()
            # Loggt ausführlichen Report
        """
        is_complete, stats = self.validate()

        logger.info("═" * 60)
        logger.info("  FKZ-basierte Index-Validierung")
        logger.info("═" * 60)
        logger.info("")
        logger.info("📊 Statistiken:")
        logger.info("   CSV-Projekte gesamt:  %d", stats["csv_total"])
        logger.info("   Indizierte Projekte:  %d", stats["indexed_total"])
        logger.info("   Synchronisation:      %.1f%%", stats["sync_percentage"])
        logger.info("")

        if stats["is_empty"]:
            logger.warning("⚠️  Index ist LEER!")
            logger.warning("   Alle %d Projekte müssen indiziert werden", stats["csv_total"])
        elif stats["new_count"] > 0:
            logger.warning("🆕 Neue Projekte:      %d", stats["new_count"])
            if stats["new_fkz"]:
                logger.info("   Beispiele: %s", ", ".join(stats["new_fkz"][:5]))
        else:
            logger.info("✅ Alle CSV-Projekte sind indiziert!")

        if stats["removed_count"] > 0:
            logger.warning("🗑️  Entfernte Projekte: %d", stats["removed_count"])
            logger.warning("   Diese sind im Index, aber nicht mehr in CSV")
            if stats["removed_fkz"]:
                logger.info("   Beispiele: %s", ", ".join(stats["removed_fkz"][:5]))

        logger.info("")
        logger.info("═" * 60)

    def get_new_projects_summary(self, limit: int = 5) -> str:
        """Erstellt lesbare Zusammenfassung neuer Projekte.

        Args:
            limit: Anzahl der anzuzeigenden Beispiele

        Returns:
            str: Formatierte Zusammenfassung
        """
        new_df = self.get_new_projects(limit=limit)

        if new_df.empty:
            return "✅ Keine neuen Projekte gefunden."

        new_count = len(self.get_new_fkz())
        summary = f"📋 {new_count} neue Projekte gefunden:\n\n"

        for idx, row in new_df.iterrows():
            fkz = row.get(self.fkz_column, "N/A")
            emp = row.get('="Zuwendungsempfänger"', "N/A")
            thema = row.get('="Thema"', "N/A")

            # Kürze lange Strings
            if len(str(emp)) > 40:
                emp = str(emp)[:37] + "..."
            if len(str(thema)) > 60:
                thema = str(thema)[:57] + "..."

            summary += f"  • FKZ {fkz}\n"
            summary += f"    {emp}\n"
            summary += f"    {thema}\n\n"

        if new_count > limit:
            summary += f"  ... und {new_count - limit} weitere Projekte\n"

        return summary


def compare_csv_files(old_csv: Path, new_csv: Path, fkz_column: str = '="FKZ"') -> dict:
    """Vergleicht zwei CSV-Dateien basierend auf FKZ.

    Nützlich um vor dem Laden zu prüfen, was sich geändert hat.

    Args:
        old_csv: Pfad zur alten CSV
        new_csv: Pfad zur neuen CSV
        fkz_column: Name der FKZ-Spalte

    Returns:
        dict: Vergleichsstatistiken

    Example:
        >>> stats = compare_csv_files(
        ...     Path("old.csv"),
        ...     Path("new.csv")
        ... )
        >>> print(f"Neue: {stats['new_count']}")
        >>> print(f"Entfernt: {stats['removed_count']}")
    """
    logger.info("Vergleiche CSV-Dateien...")
    logger.info("  Alt: %s", old_csv)
    logger.info("  Neu: %s", new_csv)

    # Lade beide CSVs
    df_old = pd.read_csv(old_csv, sep=";", encoding="latin1", low_memory=False)
    df_new = pd.read_csv(new_csv, sep=";", encoding="latin1", low_memory=False)

    # Bereinige und extrahiere FKZ
    def get_fkz_set(df):
        if fkz_column not in df.columns:
            raise ValueError(f"FKZ-Spalte '{fkz_column}' nicht gefunden")
        fkz_series = df[fkz_column].astype(str).str.strip()
        return set(fkz for fkz in fkz_series if pd.notna(fkz) and fkz not in ("", "nan", "<NA>"))

    old_fkz = get_fkz_set(df_old)
    new_fkz = get_fkz_set(df_new)

    added = new_fkz - old_fkz
    removed = old_fkz - new_fkz
    unchanged = old_fkz.intersection(new_fkz)

    stats = {
        "old_total": len(old_fkz),
        "new_total": len(new_fkz),
        "added_count": len(added),
        "removed_count": len(removed),
        "unchanged_count": len(unchanged),
        "added_fkz": sorted(list(added))[:100],
        "removed_fkz": sorted(list(removed))[:100],
        "change_percentage": (len(added) / len(old_fkz) * 100) if old_fkz else 0,
    }

    logger.info("")
    logger.info("📊 CSV-Vergleich:")
    logger.info("   Alt:        %d Projekte", stats["old_total"])
    logger.info("   Neu:        %d Projekte", stats["new_total"])
    logger.info("   Hinzugefügt: %d (%.1f%%)", stats["added_count"], stats["change_percentage"])
    logger.info("   Entfernt:   %d", stats["removed_count"])
    logger.info("   Unverändert: %d", stats["unchanged_count"])

    return stats


# ===== Convenience-Funktionen =====


def validate_with_fkz(faiss: FaissStore, df: pd.DataFrame) -> Tuple[bool, int]:
    """Convenience-Funktion für schnelle FKZ-Validierung.

    Args:
        faiss: FaissStore-Instanz
        df: DataFrame mit Projektdaten

    Returns:
        Tuple[bool, int]: (is_complete, new_count)

    Example:
        >>> is_complete, new_count = validate_with_fkz(faiss, df)
        >>> if not is_complete:
        ...     print(f"{new_count} neue Projekte müssen indiziert werden")
    """
    validator = FKZIndexValidator(faiss, df)
    validator.log_validation_report()

    is_complete, stats = validator.validate()
    return is_complete, stats["new_count"]


def get_projects_to_index(faiss: FaissStore, df: pd.DataFrame, batch_size: Optional[int] = None) -> pd.DataFrame:
    """Ermittelt Projekte, die indiziert werden müssen.

    Args:
        faiss: FaissStore-Instanz
        df: DataFrame mit allen Projekten
        batch_size: Optionales Limit für Batch-Verarbeitung

    Returns:
        pd.DataFrame: Projekte zum Indizieren

    Example:
        >>> to_index = get_projects_to_index(faiss, df, batch_size=5000)
        >>> for idx, row in to_index.iterrows():
        ...     # Indiziere Projekt
        ...     pass
    """
    validator = FKZIndexValidator(faiss, df)
    new_projects = validator.get_new_projects(limit=batch_size)

    logger.info(
        "📋 %d Projekte müssen indiziert werden%s",
        len(new_projects),
        f" (Batch-Limit: {batch_size})" if batch_size else "",
    )

    return new_projects
