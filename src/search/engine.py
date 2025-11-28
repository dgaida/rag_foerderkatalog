"""Suchmaschine für semantische und hybride Projektsuche.

Dieses Modul implementiert die Kernfunktionalität für die Suche in Förderprojekten:
- CSV-Import und Datenbereinigung
- Embedding-Erzeugung mit Ollama oder HuggingFace
- Semantische Suche via FAISS
- Kontextbasierte LLM-Antworten
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..config import INPUT_CSV, MAX_DOCS_FOR_LLM, TOP_K_DEFAULT, EmbeddingProvider
from ..embeddings.faiss_store import FaissStore
from ..llm.llm_wrapper import build_improved_user_prompt, chat_system_query, embed_text, get_improved_system_prompt
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ProjectSearchEngine:
    """Engine für semantische Suche über die Förderprojekte-CSV.

    Diese Klasse verwaltet den gesamten Suchprozess:
    - Laden und Bereinigen der CSV-Daten
    - Erzeugung und Verwaltung von Embeddings (Ollama oder HuggingFace)
    - Semantische Suche via FAISS
    - Kontextbasierte Antwortgenerierung

    Attributes:
        csv_file: Pfad zur CSV-Datei mit Förderprojekten.
        df: Pandas DataFrame mit den geladenen Projektdaten.
        provider: Embedding-Provider ("ollama" oder "huggingface")
        embed_model: Name des Embedding-Modells
        faiss: FaissStore-Instanz für Vektorsuche.

    Example:
        >>> # Mit Ollama (Default)
        >>> engine = ProjectSearchEngine()
        >>> engine.load_and_clean()
        >>> results = engine.search("Künstliche Intelligenz", k=10)

        >>> # Mit HuggingFace
        >>> engine = ProjectSearchEngine(
        ...     provider="huggingface",
        ...     embed_model="intfloat/e5-small-v2"
        ... )
        >>> engine.load_and_clean()
        >>> results = engine.search("Künstliche Intelligenz", k=10)
    """

    def __init__(
        self,
        csv_file: Optional[Path] = None,
        provider: EmbeddingProvider = "ollama",
        embed_model: Optional[str] = None,
    ):
        """Initialisiert die Search Engine.

        Args:
            csv_file: Optionaler Pfad zur CSV-Datei. Wenn None, wird INPUT_CSV verwendet.
            provider: Embedding-Provider ("ollama" oder "huggingface")
            embed_model: Name des Embedding-Modells. Wenn None, wird Default verwendet.
        """
        self.csv_file = Path(csv_file) if csv_file else INPUT_CSV
        self.df: Optional[pd.DataFrame] = None
        self.provider = provider
        self.embed_model = embed_model
        self.faiss = FaissStore(provider=provider)

        logger.info(
            "SearchEngine initialisiert mit Provider: %s, Modell: %s",
            provider,
            embed_model or "default",
        )

    def _extract_year(self, date_str: str) -> Optional[int]:
        """Extrahiert die Jahreszahl aus einem Datumsstring.

        Args:
            date_str: Datumsstring im Format DD.MM.YYYY oder ähnlich.

        Returns:
            Optional[int]: Extrahierte Jahreszahl oder None bei Fehler.

        Example:
            >>> self._extract_year("01.03.2002")
            2002
        """
        if pd.isna(date_str) or not isinstance(date_str, str):
            return None

        # Suche nach 4-stelliger Jahreszahl
        match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if match:
            try:
                return int(match.group(0))
            except ValueError:
                return None
        return None

    def _create_runtime_string(self, row: pd.Series) -> str:
        """Erstellt einen Laufzeit-String aus Start- und Enddatum.

        Args:
            row: Pandas Series mit 'Laufzeit von' und 'Laufzeit bis' Spalten.

        Returns:
            str: Formatierter Laufzeit-String (z.B. "2002 - 2005") oder leerer String.

        Example:
            >>> row = pd.Series({'Laufzeit von': '01.03.2002', 'Laufzeit bis': '31.12.2005'})
            >>> self._create_runtime_string(row)
            "2002 - 2005"
        """
        start_year = self._extract_year(row.get('="Laufzeit von"', ""))
        end_year = self._extract_year(row.get('="Laufzeit bis"', ""))

        if start_year and end_year:
            return f"{start_year} - {end_year}"
        elif start_year:
            return f"ab {start_year}"
        elif end_year:
            return f"bis {end_year}"
        return ""

    def load_and_clean(self) -> None:
        """Lädt die CSV-Datei und führt Basisbereinigungen durch.

        Diese Methode:
        - Lädt die CSV mit korrektem Encoding (latin1)
        - Entfernt Excel-Formatierungszeichen (=", ")
        - Konvertiert Fördersummen in numerische Werte
        - Erstellt eine Laufzeit-Spalte

        Raises:
            FileNotFoundError: Wenn die CSV-Datei nicht existiert.
            Exception: Bei Fehlern während des Ladens oder Reinigens.
        """
        logger.info("Lade CSV: %s", self.csv_file)

        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV-Datei nicht gefunden: {self.csv_file}")

        df = pd.read_csv(self.csv_file, sep=";", encoding="latin1", low_memory=False)

        # Entferne Excel-Formatierung von allen String-Spalten
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace('="', "", regex=False).str.replace('"', "", regex=False)

        # Bereinige Fördersumme und konvertiere zu numerisch
        if '="Fördersumme in EUR"' in df.columns:
            df['="Fördersumme in EUR"'] = (
                df['="Fördersumme in EUR"'].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            )
            df['="Fördersumme in EUR"'] = pd.to_numeric(df['="Fördersumme in EUR"'], errors="coerce")

        # Erstelle Laufzeit-Spalte
        df["__laufzeit"] = df.apply(self._create_runtime_string, axis=1)

        self.df = df
        logger.info("CSV geladen: %d Zeilen, %d Spalten", df.shape[0], df.shape[1])
        logger.info("Laufzeit-Spalte erstellt mit %d gültigen Einträgen", (df["__laufzeit"] != "").sum())

    def _build_embedding_text(self, row: pd.Series) -> str:
        """Erstellt den Text für Embedding-Erzeugung aus einer Projektzeile.

        Kombiniert relevante Spalten inkl. Laufzeit zu einem durchsuchbaren Text.

        Args:
            row: Pandas Series mit Projektdaten.

        Returns:
            str: Zusammengesetzter Text für Embedding.
        """
        # Erweiterte Spaltenliste für umfassendere Embeddings
        text_fields = [
            '="Zuwendungsempfänger"',
            '="Thema"',
            '="Klartext Leistungsplansystematik"',
            '="Ausführende Stelle"',
            '="Stadt/Gemeinde"',
            '="Bundesland"',
            "__laufzeit",  # Neu: Laufzeit-Information
            '="Förderprofil"',
            '="Verbundprojekt"',
        ]

        parts = []
        for field in text_fields:
            if field in row.index:
                val = row.get(field)
                if pd.notna(val) and str(val).strip() and str(val) != "nan":
                    parts.append(str(val))

        return ". ".join(parts) if parts else ""

    def build_embeddings_if_missing(self, text_fields: Optional[List[str]] = None) -> None:
        """Erstellt Embeddings für alle Projekte, falls FAISS-Index leer ist.

        Args:
            text_fields: Optionale Liste von Spalten für Embedding-Text.
                        Wenn None, werden Default-Felder verwendet.

        Raises:
            RuntimeError: Wenn DataFrame nicht geladen ist.
            Exception: Bei Fehlern während der Embedding-Erzeugung.
        """
        if self.df is None:
            raise RuntimeError("Dataframe nicht geladen. Rufe load_and_clean() auf.")

        if self.faiss.index is not None and self.faiss.index.ntotal > 0:
            logger.info(
                "FAISS (%s) enthält bereits %d Vektoren, überspringe Embedding-Erstellung.",
                self.provider,
                self.faiss.index.ntotal,
            )
            return

        logger.info(
            "Erzeuge Embeddings für %d Dokumente mit %s (kann lange dauern)...",
            len(self.df),
            self.provider,
        )

        for idx, row in self.df.iterrows():
            text = self._build_embedding_text(row)

            if not text:
                logger.warning("Leerer Text für Zeile %s, überspringe", idx)
                continue

            try:
                vec = embed_text(text, provider=self.provider, model=self.embed_model)
                self.faiss.add(vec, doc_id=str(idx), persist_now=False)

                if (idx + 1) % 100 == 0:
                    logger.info("Embeddings: %d/%d verarbeitet", idx + 1, len(self.df))

            except Exception as e:
                logger.exception("Embedding fehlgeschlagen für Zeile %s: %s", idx, e)

        self.faiss.persist()
        logger.info("Embeddings persistiert (%s).", self.provider)

    def search(self, query: str, k: int = TOP_K_DEFAULT) -> pd.DataFrame:
        """Führt eine semantische Suche durch.

        Args:
            query: Suchanfrage als String.
            k: Anzahl der zurückzugebenden Treffer.

        Returns:
            pd.DataFrame: Gefundene Projekte mit Score-Spalte, sortiert nach Relevanz.

        Raises:
            RuntimeError: Wenn DataFrame nicht geladen ist.
            ValueError: Bei Dimensions-Mismatch zwischen Index und Query.
        """
        if self.df is None:
            raise RuntimeError("Dataframe nicht geladen.")

        try:
            qvec = embed_text(query, provider=self.provider, model=self.embed_model)

            if self.faiss.index is None:
                logger.error("FAISS Index ist leer oder nicht initialisiert.")
                return pd.DataFrame()

            dim_index = self.faiss.index.d
            dim_query = len(qvec)

            if dim_index != dim_query:
                error_msg = (
                    f"Embedding-Dimension stimmt nicht überein! "
                    f"Bitte Index löschen und Embeddings neu erzeugen. "
                    f"(Index={dim_index}, Query={dim_query})"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            hits = self.faiss.search(qvec, k=k)
            indices = [int(doc_id) for _, doc_id in hits]
            results = self.df.iloc[indices].copy()
            scores = [score for score, _ in hits]
            results["__score"] = scores
            results = results.sort_values("__score", ascending=False)

            logger.info(
                "Semantische Suche (%s) nach '%s' lieferte %d Treffer",
                self.provider,
                query,
                len(results),
            )
            return results

        except Exception as e:
            logger.exception("Fehler während Suche: %s", e)
            return pd.DataFrame()

    def answer_with_context(self, query: str) -> str:
        """Generiert eine kontextbasierte LLM-Antwort auf die Suchanfrage.

        Args:
            query: Nutzeranfrage als String.

        Returns:
            str: Vom LLM generierte Antwort basierend auf relevanten Projekten.

        Raises:
            Exception: Bei Fehlern während der LLM-Kommunikation.
        """
        results = self.search(query, k=MAX_DOCS_FOR_LLM)

        if results.empty:
            return "Keine relevanten Projekte gefunden."

        # Erstelle detaillierte Snippets
        snippets = []
        for i, (_, row) in enumerate(results.head(MAX_DOCS_FOR_LLM).iterrows(), 1):
            fkz = row.get('="FKZ"', "N/A")
            emp = row.get('="Zuwendungsempfänger"', "N/A")
            thema = row.get('="Thema"', "N/A")
            summe = row.get('="Fördersumme in EUR"', "N/A")
            laufzeit = row.get("__laufzeit", "N/A")
            bundesland = row.get('="Bundesland"', "N/A")

            snippet = (
                f"{i}. **FKZ**: {fkz}\n"
                f"   **Empfänger**: {emp}\n"
                f"   **Thema**: {thema}\n"
                f"   **Fördersumme**: {summe} EUR\n"
                f"   **Laufzeit**: {laufzeit}\n"
                f"   **Bundesland**: {bundesland}"
            )
            snippets.append(snippet)

        system_prompt = get_improved_system_prompt()
        user_prompt = build_improved_user_prompt(snippets, query)

        answer = chat_system_query(system_prompt, user_prompt)
        return answer

    def get_index_info(self) -> dict:
        """Gibt Informationen über den aktuellen Index zurück.

        Returns:
            dict: Dictionary mit Index-Informationen
        """
        info = self.faiss.get_info()
        info["csv_loaded"] = self.df is not None
        info["csv_rows"] = len(self.df) if self.df is not None else 0
        return info
