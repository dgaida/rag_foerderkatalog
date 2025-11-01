# src/llm/llm_wrapper.py
"""LLM-Wrapper für Embeddings und Chat-Completion.

Dieses Modul stellt Funktionen für die Interaktion mit LLMs bereit:
- Embedding-Erzeugung via Ollama
- Chat-Completion via LLMClient
- Prompt-Persistierung für Debugging
"""

from typing import List, Optional
import logging
from pathlib import Path
from datetime import datetime
from llm_client import LLMClient
from ollama import embed as ollama_embed

logger = logging.getLogger(__name__)


def save_prompt_to_md(prompt: str, folder: str = "logs/prompts") -> Path:
    """Speichert den vollständigen Prompt als Markdown-Datei mit Zeitstempel.

    Args:
        prompt: Der vollständige Text des Prompts, der gespeichert werden soll.
        folder: Zielverzeichnis für die gespeicherte Datei. Defaults to "logs/prompts".

    Returns:
        Path: Der vollständige Pfad zur erstellten Markdown-Datei.

    Raises:
        OSError: Wenn das Verzeichnis nicht erstellt werden kann.
    """
    Path(folder).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = Path(folder) / f"prompt-{timestamp}.md"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# LLM Prompt ({timestamp})\n\n")
            f.write(prompt)
        logger.info("Prompt gespeichert: %s", filename)
    except Exception as e:
        logger.error("Fehler beim Speichern des Prompts: %s", e)
        raise
    return filename


def chat_system_query(
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None
) -> str:
    """Führt eine Chat-Anfrage an ein LLM über LLMClient aus.

    Args:
        system_prompt: System-Prompt zur Steuerung des Modellverhaltens.
        user_prompt: Nutzerprompt mit der eigentlichen Anfrage.
        model: Modellname für LLMClient. Wenn None, wird das konfigurierte Default verwendet.

    Returns:
        str: Generierte Antwort des LLMs.

    Raises:
        Exception: Bei Fehlern während der LLM-Kommunikation.

    Example:
        >>> answer = chat_system_query(
        ...     "Du bist ein hilfsbereiter Assistent.",
        ...     "Was ist Künstliche Intelligenz?"
        ... )
    """
    save_prompt_to_md(user_prompt)

    try:
        client = LLMClient(llm=model, max_tokens=1024, temperature=0.5)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = client.chat_completion(messages)
        return resp
    except Exception as e:
        logger.exception("Fehler beim LLM-Chat: %s", e)
        raise


def embed_text(text: str, model: str = "nomic-embed-text") -> List[float]:
    """Erstellt ein Embedding mittels Ollama, robust gegen verschiedene API-Rückgabeformate.

    Args:
        text: Der zu embeddierende Text.
        model: Name des Ollama-Embedding-Modells. Defaults to "nomic-embed-text".

    Returns:
        List[float]: Normalisierter Embedding-Vektor als Liste von Floats.

    Raises:
        TypeError: Wenn der Rückgabewert von Ollama ein unerwartetes Format hat.
        ValueError: Wenn der Embedding-Vektor ungültig ist.
        Exception: Bei sonstigen Fehlern während der Embedding-Erzeugung.

    Example:
        >>> vec = embed_text("Künstliche Intelligenz Forschung")
        >>> len(vec)
        768
    """
    try:
        resp = ollama_embed(model=model, input=text)

        # Neuere Ollama-Versionen -> EmbedResponse-Objekt
        if hasattr(resp, "embeddings"):
            emb = resp.embeddings
        # Ältere Version -> Dictionary
        elif isinstance(resp, dict):
            emb = resp.get("embeddings") or resp.get("embedding")
        else:
            raise TypeError(f"Unerwarteter Rückgabewert: {type(resp)} -> {resp}")

        # Extrahiere tatsächlichen Vektor
        if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
            emb = emb[0]

        # Validierung
        if not isinstance(emb, list) or not all(isinstance(x, (float, int)) for x in emb):
            raise ValueError(f"Embedding-Vektor hat unerwartetes Format: {type(emb)}")

        return [float(x) for x in emb]

    except Exception as e:
        logger.exception("Fehler bei Ollama embed: %s", e)
        raise


def get_improved_system_prompt() -> str:
    """Liefert einen verbesserten System-Prompt für RAG-basierte Antworten.

    Returns:
        str: Optimierter System-Prompt.
    """
    return """Du bist ein spezialisierter KI-Assistent für deutsche Forschungsförderung und analysierst Förderprojekte des BMBF (Bundesministerium für Bildung und Forschung).

**Deine Aufgaben:**
1. Beantworte Fragen präzise auf Basis der bereitgestellten Projekt-Snippets
2. Nenne immer die relevanten Förderkennzeichen (FKZ) als Quellenangabe
3. Fasse mehrere Projekte zusammen, wenn sie thematisch zusammengehören
4. Nutze Fördersummen und Laufzeiten für quantitative Aussagen

**Wichtige Regeln:**
- Beziehe dich ausschließlich auf die bereitgestellten Informationen
- Spekuliere nicht und erfinde keine Daten
- Wenn Informationen fehlen, sage das explizit
- Strukturiere längere Antworten mit Absätzen
- Verwende präzise Fachbegriffe aus der Forschungsförderung

**Ausgabeformat:**
- Beginne mit einer knappen Zusammenfassung (1-2 Sätze)
- Liste relevante Projekte mit FKZ auf. Beginne jedes Element der Liste mit: "- **FKZ: [...]**". [...] ist ein Platzhalte für das FKZ, bspw.: "03KB045A".
- Schließe mit statistischen Eckdaten ab (Anzahl Projekte, Gesamtfördersumme, Zeitraum)"""


def build_improved_user_prompt(snippets: List[str], query: str) -> str:
    """Erstellt einen optimierten User-Prompt für kontextbasierte LLM-Antworten.

    Args:
        snippets: Liste von Projekt-Snippets als Kontext.
        query: Die Nutzeranfrage.

    Returns:
        str: Formatierter User-Prompt.
    """
    context_block = "\n".join(snippets)

    return f"""**KONTEXT (Förderprojekte):**
{context_block}

**NUTZERANFRAGE:**
{query}

**ANWEISUNG:**
Beantworte die Nutzeranfrage ausschließlich basierend auf den obigen Projekt-Snippets. Strukturiere deine Antwort wie folgt:

1. **Zusammenfassung** (2-3 Sätze): Kernaussage zur Anfrage
2. **Relevante Projekte**: Liste mit FKZ, Empfänger, Thema
3. **Quantitative Analyse**: Anzahl Projekte, Gesamtfördersumme, Zeitraum
4. **Fazit** (1-2 Sätze): Einordnung der Ergebnisse

Beginne jetzt mit deiner Antwort:"""
