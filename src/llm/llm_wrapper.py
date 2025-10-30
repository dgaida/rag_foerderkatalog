# src/llm/llm_wrapper.py
from typing import List
import logging
from pathlib import Path
from datetime import datetime
from llm_client import LLMClient # external dependency (see README)
from ollama import embed as ollama_embed

logger = logging.getLogger(__name__)


def save_prompt_to_md(prompt: str, folder: str = "logs/prompts") -> Path:
    """Speichert den vollständigen Prompt als Markdown-Datei mit Zeitstempel.

    Erstellt das angegebene Zielverzeichnis (falls nicht vorhanden) und legt dort
    eine Markdown-Datei mit dem vollständigen Prompt-Inhalt ab. Der Dateiname
    enthält einen eindeutigen Zeitstempel.

    Args:
        prompt (str): Der vollständige Text des Prompts, der gespeichert werden soll.
        folder (str, optional): Zielverzeichnis für die gespeicherte Datei.
            Standardmäßig `"logs/prompts"`.

    Returns:
        Path: Der vollständige Pfad zur erstellten Markdown-Datei.
    """
    Path(folder).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = Path(folder) / f"prompt-{timestamp}.md"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# LLM Prompt ({timestamp})\n\n")
            f.write(prompt)
        logger.info(f"Prompt gespeichert: {filename}")
    except Exception as e:
        logger.error(f"Fehler beim Speichern des Prompts: {e}")
    return filename


def chat_system_query(system_prompt: str, user_prompt: str, model: str = None) -> str:
    """Führt eine Chat-Anfrage an ein LLM über LLMClient aus.

    Args:
        system_prompt: System-Prompt zur Steuerung des Modells.
        user_prompt: Nutzerprompt.
        model: Modellname für LLMClient (falls None, konfiguriertes Default verwenden).

    Returns:
        Generierte Antwort als String.
    """
    save_prompt_to_md(user_prompt)

    try:
        client = LLMClient(llm=model)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = client.chat_completion(messages)
        return resp
    except Exception as e:
        logger.exception("Fehler beim LLM-Chat")
        raise


def embed_text(text: str, model: str = "nomic-embed-text") -> list[float]:
    """
    Erstellt ein Embedding mittels Ollama `embed`, robust gegen alle API-Rückgabeformate.
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
