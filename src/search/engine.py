from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path
import logging

from ..utils.logging_config import get_logger
from ..embeddings.faiss_store import FaissStore
from ..llm.llm_wrapper import embed_text, chat_system_query
from ..config import INPUT_CSV, EMBEDDINGS_FILE, TOP_K_DEFAULT, MAX_DOCS_FOR_LLM

logger = get_logger(__name__)


class ProjectSearchEngine:
    """Engine für semantische Suche über die Förderprojekte-CSV.

    Embeddings werden mit Ollama erzeugt (via `embed_text`) und in FAISS persistiert (via `FaissStore`).

    Teile dieser Klasse basieren auf dem Projekt:
    https://github.com/ibaleri/Foerderprojekt
    """

    def __init__(self, csv_file: Optional[Path] = None):
        self.csv_file = Path(csv_file) if csv_file else INPUT_CSV
        self.df: Optional[pd.DataFrame] = None
        self.faiss = FaissStore()

    def load_and_clean(self) -> None:
        """Lädt die CSV-Datei und führt Basisbereinigungen durch."""
        logger.info("Lade CSV: %s", self.csv_file)
        df = pd.read_csv(self.csv_file, sep=';', encoding='latin1', low_memory=False)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace('="', '', regex=False).str.replace('"', '', regex=False)

        if 'Fördersumme in EUR' in df.columns:
            df['Fördersumme in EUR'] = df['Fördersumme in EUR'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df['Fördersumme in EUR'] = pd.to_numeric(df['Fördersumme in EUR'], errors='coerce')

        self.df = df
        logger.info("CSV geladen: %d Zeilen, %d Spalten", df.shape[0], df.shape[1])

    def build_embeddings_if_missing(self, text_fields: Optional[List[str]] = None) -> None:
        if self.df is None:
            raise RuntimeError("Dataframe nicht geladen. Rufe load_and_clean() auf.")

        text_fields = text_fields or ['Zuwendungsempfänger', 'Thema', 'Klartext Leistungsplansystematik']
        if self.faiss.index is not None and self.faiss.index.ntotal > 0:
            logger.info("FAISS enthält bereits %d Vektoren, überspringe Embedding-Erstellung.", self.faiss.index.ntotal)
            return

        logger.info("Erzeuge Embeddings für %d Dokumente (kann lange dauern)...", len(self.df))
        for idx, row in self.df.iterrows():
            parts = []
            for col in text_fields:
                if col in self.df.columns:
                    val = row.get(col)
                    if pd.notna(val):
                        parts.append(str(val))
            text = '. '.join(parts)
            try:
                vec = embed_text(text)
                self.faiss.add(vec, doc_id=str(idx), persist_now=False)
            except Exception as e:
                logger.exception("Embedding fehlgeschlagen für Zeile %s: %s", idx, e)
        self.faiss.persist()
        logger.info("Embeddings persistiert.")

    def search(self, query: str, k: int = TOP_K_DEFAULT) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Dataframe nicht geladen.")

        try:
            qvec = embed_text(query)

            # 🧩 Debug-Ausgabe und Dim-Konsistenzprüfung
            if self.faiss.index is None:
                logger.error("FAISS Index ist leer oder nicht initialisiert.")
                return pd.DataFrame()

            dim_index = self.faiss.index.d
            dim_query = len(qvec)
            if dim_index != dim_query:
                logger.error(f"Dimension mismatch: Index={dim_index}, Query={dim_query}.")
                raise ValueError(
                    f"Embedding-Dimension stimmt nicht überein! "
                    f"Bitte Index löschen und Embeddings neu erzeugen. "
                    f"(Index={dim_index}, Query={dim_query})"
                )

            hits = self.faiss.search(qvec, k=k)
            indices = [int(doc_id) for _, doc_id in hits]
            results = self.df.iloc[indices].copy()
            scores = [score for score, _ in hits]
            results['__score'] = scores
            results = results.sort_values('__score', ascending=False)
            return results

        except Exception as e:
            logger.exception("Fehler während Suche: %s", e)
            return pd.DataFrame()

    def answer_with_context(self, query: str) -> str:
        results = self.search(query, k=MAX_DOCS_FOR_LLM)
        if results.empty:
            return "Keine relevanten Projekte gefunden."

        snippets = []
        for i, (_, row) in enumerate(results.head(MAX_DOCS_FOR_LLM).iterrows()):
            fkz = row.get('FKZ', '')
            emp = row.get('Zuwendungsempfänger', '')
            thema = row.get('Thema', '')
            summe = row.get('Fördersumme in EUR', '')
            snippets.append(f"{i+1}. FKZ: {fkz} | Empfänger: {emp} | Thema: {thema} | Fördersumme: {summe}")

        system_prompt = "Du bist ein Hilfs-LLM, das Antworten basierend auf gelieferten Projekt-Snippets gibt. Verwende nur die vorliegenden Informationen."
        user_prompt = f"Kontext:\n{chr(10).join(snippets)}\n\nFrage: {query}\nGib eine kurze, belegte Antwort und nenne die relevanten FKZ(s)."

        answer = chat_system_query(system_prompt, user_prompt)
        return answer
