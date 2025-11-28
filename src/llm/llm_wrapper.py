# src/llm/llm_wrapper.py
"""LLM-Wrapper für Embeddings und Chat-Completion.

Dieses Modul stellt Funktionen für die Interaktion mit LLMs bereit:
- Embedding-Erzeugung via Ollama oder HuggingFace
- Chat-Completion via LLMClient
- Prompt-Persistierung für Debugging
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from llm_client import LLMClient
from ollama import embed as ollama_embed

logger = logging.getLogger(__name__)

# Lazy-Import für HuggingFace (nur wenn benötigt)
_hf_embed_model = None
_current_hf_model_name = None


def _get_hf_embedding_model(model_name: str):
    """Lazy-Loading für HuggingFace Embedding-Modell.

    Args:
        model_name: Name des HuggingFace-Modells

    Returns:
        HuggingFaceEmbedding-Instanz
    """
    global _hf_embed_model, _current_hf_model_name

    # Wenn bereits geladen und gleiches Modell, wiederverwenden
    if _hf_embed_model is not None and _current_hf_model_name == model_name:
        return _hf_embed_model

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        logger.info("Lade HuggingFace Embedding-Modell: %s", model_name)
        _hf_embed_model = HuggingFaceEmbedding(model_name=model_name)
        _current_hf_model_name = model_name
        logger.info("HuggingFace Modell erfolgreich geladen")

        return _hf_embed_model
    except ImportError as e:
        logger.error(
            "llama-index-embeddings-huggingface nicht installiert. "
            "Installieren Sie mit: pip install llama-index-embeddings-huggingface"
        )
        raise ImportError(
            "HuggingFace Embeddings benötigen llama-index-embeddings-huggingface. "
            "Installieren Sie mit: pip install llama-index-embeddings-huggingface"
        ) from e


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


def chat_system_query(system_prompt: str, user_prompt: str, model: Optional[str] = None) -> str:
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


def embed_text(text: str, provider: Literal["ollama", "huggingface"] = "ollama", model: Optional[str] = None) -> List[float]:
    """Erstellt ein Embedding mittels Ollama oder HuggingFace.

    Args:
        text: Der zu embeddierende Text.
        provider: "ollama" oder "huggingface". Defaults to "ollama".
        model: Modellname. Wenn None, wird Default verwendet.
            - Ollama: "nomic-embed-text"
            - HuggingFace: "intfloat/e5-small-v2"

    Returns:
        List[float]: Normalisierter Embedding-Vektor als Liste von Floats.

    Raises:
        ValueError: Wenn der Embedding-Vektor ungültig ist oder Provider unbekannt.
        ImportError: Wenn HuggingFace-Dependencies fehlen.
        Exception: Bei sonstigen Fehlern während der Embedding-Erzeugung.

    Example:
        >>> # Ollama
        >>> vec_ollama = embed_text("KI Forschung", provider="ollama")
        >>> len(vec_ollama)
        768

        >>> # HuggingFace
        >>> vec_hf = embed_text("KI Forschung", provider="huggingface")
        >>> len(vec_hf)
        384
    """
    if provider == "ollama":
        return _embed_text_ollama(text, model or "nomic-embed-text")
    elif provider == "huggingface":
        return _embed_text_huggingface(text, model or "intfloat/e5-small-v2")
    else:
        raise ValueError(f"Unbekannter Provider: {provider}. Verwenden Sie 'ollama' oder 'huggingface'.")


def _embed_text_ollama(text: str, model: str) -> List[float]:
    """Erstellt Embedding via Ollama.

    Args:
        text: Zu embeddierender Text.
        model: Ollama-Modellname.

    Returns:
        List[float]: Embedding-Vektor.
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


def _embed_text_huggingface(text: str, model_name: str) -> List[float]:
    """Erstellt Embedding via HuggingFace.

    Args:
        text: Zu embeddierender Text.
        model_name: HuggingFace-Modellname.

    Returns:
        List[float]: Embedding-Vektor.
    """
    try:
        embed_model = _get_hf_embedding_model(model_name)

        # LlamaIndex HuggingFaceEmbedding hat get_text_embedding() Methode
        embedding = embed_model.get_text_embedding(text)

        # Validierung
        if not isinstance(embedding, list) or not all(isinstance(x, (float, int)) for x in embedding):
            raise ValueError(f"HuggingFace Embedding hat unerwartetes Format: {type(embedding)}")

        return [float(x) for x in embedding]

    except Exception as e:
        logger.exception("Fehler bei HuggingFace embed: %s", e)
        raise


def get_embedding_dimension(provider: Literal["ollama", "huggingface"], model: Optional[str] = None) -> int:
    """Ermittelt die Embedding-Dimension für ein Modell.

    Args:
        provider: "ollama" oder "huggingface"
        model: Modellname (optional, verwendet Default wenn None)

    Returns:
        int: Dimension des Embedding-Vektors

    Example:
        >>> get_embedding_dimension("ollama", "nomic-embed-text")
        768
        >>> get_embedding_dimension("huggingface", "intfloat/e5-small-v2")
        384
    """
    # Erstelle Test-Embedding und messe Dimension
    test_vec = embed_text("test", provider=provider, model=model)
    return len(test_vec)


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
