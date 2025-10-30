# src/llm/llm_wrapper.py
from typing import List
import logging
from llm_client import LLMClient # external dependency (see README)
from ollama import embed as ollama_embed

logger = logging.getLogger(__name__)


def chat_system_query(system_prompt: str, user_prompt: str, model: str = None) -> str:
    """Führt eine Chat-Anfrage an ein LLM über LLMClient aus.

    Args:
        system_prompt: System-Prompt zur Steuerung des Modells.
        user_prompt: Nutzerprompt.
        model: Modellname für LLMClient (falls None, konfiguriertes Default verwenden).

    Returns:
        Generierte Antwort als String.
    """
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
