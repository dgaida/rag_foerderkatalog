"""Unit-Tests für src/llm/llm_wrapper.py

Tests für:
- Prompt-Persistierung
- Embedding-Erzeugung
- Chat-Completion
- Prompt-Generierung
- HuggingFace Embedding-Provider
- Embedding-Dimension-Erkennung
- Error-Handling
- Provider-Switching
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.llm.llm_wrapper import (
    save_prompt_to_md,
    chat_system_query,
    embed_text,
    get_improved_system_prompt,
    build_improved_user_prompt,
    get_embedding_dimension,
)


class TestHuggingFaceEmbeddings:
    """Tests für HuggingFace Embedding-Provider."""

    @patch("src.llm.llm_wrapper._get_hf_embedding_model")
    def test_embed_text_huggingface_success(self, mock_get_model):
        """Test: HuggingFace Embeddings funktionieren."""
        mock_model = MagicMock()
        mock_model.get_text_embedding.return_value = [0.1, 0.2, 0.3]
        mock_get_model.return_value = mock_model

        result = embed_text("test text", provider="huggingface")

        assert result == [0.1, 0.2, 0.3]
        mock_model.get_text_embedding.assert_called_once_with("test text")

    @patch("src.llm.llm_wrapper._get_hf_embedding_model")
    def test_embed_text_huggingface_with_custom_model(self, mock_get_model):
        """Test: Custom HuggingFace-Modell wird verwendet."""
        mock_model = MagicMock()
        mock_model.get_text_embedding.return_value = [1.0] * 384
        mock_get_model.return_value = mock_model

        embed_text("test", provider="huggingface", model="intfloat/e5-small-v2")

        mock_get_model.assert_called_once_with("intfloat/e5-small-v2")

    def test_embed_text_invalid_provider_raises_error(self):
        """Test: Ungültiger Provider führt zu ValueError."""
        with pytest.raises(ValueError, match="Unbekannter Provider"):
            embed_text("test", provider="invalid_provider")

    @patch("src.llm.llm_wrapper._get_hf_embedding_model")
    def test_embed_text_huggingface_invalid_format_raises_error(self, mock_get_model):
        """Test: Ungültiges Embedding-Format führt zu ValueError."""
        mock_model = MagicMock()
        mock_model.get_text_embedding.return_value = "invalid_format"
        mock_get_model.return_value = mock_model

        with pytest.raises(ValueError, match="unerwartetes Format"):
            embed_text("test", provider="huggingface")


class TestGetHuggingFaceModel:
    """Tests für HuggingFace Modell-Lazy-Loading."""

    @patch("src.llm.llm_wrapper.HuggingFaceEmbedding", create=True)
    def test_get_hf_model_lazy_loading(self, mock_hf_class):
        """Test: HuggingFace-Modell wird lazy geladen."""
        from src.llm.llm_wrapper import _get_hf_embedding_model

        # Reset global state
        import src.llm.llm_wrapper as llm_module

        llm_module._hf_embed_model = None
        llm_module._current_hf_model_name = None

        mock_model = MagicMock()
        mock_hf_class.return_value = mock_model

        # Verwende einen Mock-Modellnamen statt einen echten
        result = _get_hf_embedding_model("mock-test-model")

        assert result == mock_model
        mock_hf_class.assert_called_once_with(model_name="mock-test-model")

    @patch("src.llm.llm_wrapper.HuggingFaceEmbedding", create=True)
    def test_get_hf_model_caching(self, mock_hf_class):
        """Test: HuggingFace-Modell wird gecacht."""
        from src.llm.llm_wrapper import _get_hf_embedding_model

        # Reset global state
        import src.llm.llm_wrapper as llm_module

        llm_module._hf_embed_model = None
        llm_module._current_hf_model_name = None

        mock_model = MagicMock()
        mock_hf_class.return_value = mock_model

        # Verwende Mock-Modellnamen
        # Erstes Laden
        result1 = _get_hf_embedding_model("mock-same-model")
        # Zweites Laden (sollte Cache verwenden)
        result2 = _get_hf_embedding_model("mock-same-model")

        assert result1 == result2
        # Sollte nur einmal aufgerufen werden (Caching)
        mock_hf_class.assert_called_once()

    def test_get_hf_model_missing_import_raises_error(self):
        """Test: ImportError wenn HuggingFace nicht installiert."""
        from src.llm.llm_wrapper import _get_hf_embedding_model

        with patch.dict("sys.modules", {"llama_index.embeddings.huggingface": None}):
            # Trigger ImportError durch fehlende llama_index
            with pytest.raises(ImportError, match="llama-index-embeddings-huggingface"):
                _get_hf_embedding_model("test-model")


class TestGetEmbeddingDimension:
    """Tests für get_embedding_dimension()."""

    @patch("src.llm.llm_wrapper.embed_text")
    def test_get_dimension_ollama(self, mock_embed):
        """Test: Dimension für Ollama-Modell wird ermittelt."""
        mock_embed.return_value = [0.1] * 768

        dim = get_embedding_dimension("ollama", "nomic-embed-text")

        assert dim == 768
        mock_embed.assert_called_once_with("test", provider="ollama", model="nomic-embed-text")

    @patch("src.llm.llm_wrapper.embed_text")
    def test_get_dimension_huggingface(self, mock_embed):
        """Test: Dimension für HuggingFace-Modell wird ermittelt."""
        mock_embed.return_value = [0.1] * 384

        dim = get_embedding_dimension("huggingface", "intfloat/e5-small-v2")

        assert dim == 384

    @patch("src.llm.llm_wrapper.embed_text")
    def test_get_dimension_default_model(self, mock_embed):
        """Test: Dimension mit Default-Modell."""
        mock_embed.return_value = [0.1] * 512

        dim = get_embedding_dimension("ollama")

        assert dim == 512
        # Sollte mit model=None aufgerufen werden
        mock_embed.assert_called_once_with("test", provider="ollama", model=None)


class TestOllamaEmbeddingEdgeCases:
    """Tests für Edge-Cases bei Ollama Embeddings."""

    @patch("src.llm.llm_wrapper.ollama_embed")
    def test_embed_ollama_with_dict_and_embedding_key(self, mock_embed):
        """Test: Ollama-Response mit 'embedding' Key (statt 'embeddings')."""
        mock_embed.return_value = {"embedding": [1.0, 2.0, 3.0]}

        result = embed_text("test", provider="ollama")

        assert result == [1.0, 2.0, 3.0]

    @patch("src.llm.llm_wrapper.ollama_embed")
    def test_embed_ollama_with_nested_list(self, mock_embed):
        """Test: Ollama-Response mit verschachtelter Liste."""
        mock_embed.return_value = {"embeddings": [[1.5, 2.5, 3.5]]}

        result = embed_text("test", provider="ollama")

        # Sollte die innere Liste extrahieren
        assert result == [1.5, 2.5, 3.5]

    @patch("src.llm.llm_wrapper.ollama_embed")
    def test_embed_ollama_converts_to_float(self, mock_embed):
        """Test: Integer-Werte werden zu Float konvertiert."""
        mock_embed.return_value = {"embeddings": [[1, 2, 3]]}

        result = embed_text("test", provider="ollama")

        assert all(isinstance(x, float) for x in result)
        assert result == [1.0, 2.0, 3.0]

    @patch("src.llm.llm_wrapper.ollama_embed")
    def test_embed_ollama_exception_is_raised(self, mock_embed):
        """Test: Exceptions werden weitergeleitet."""
        mock_embed.side_effect = RuntimeError("Ollama connection error")

        with pytest.raises(RuntimeError, match="Ollama connection error"):
            embed_text("test", provider="ollama")


class TestPromptFunctions:
    """Tests für Prompt-Generierungs-Funktionen."""

    def test_system_prompt_contains_keywords(self):
        """Test: System-Prompt enthält wichtige Keywords."""
        from src.llm.llm_wrapper import get_improved_system_prompt

        prompt = get_improved_system_prompt()

        keywords = ["KI-Assistent", "BMBF", "Förderkennzeichen", "FKZ", "Regeln", "Aufgaben"]
        for keyword in keywords:
            assert keyword in prompt

    def test_user_prompt_includes_all_snippets(self):
        """Test: User-Prompt enthält alle Snippets."""
        from src.llm.llm_wrapper import build_improved_user_prompt

        snippets = ["Snippet 1: FKZ ABC123", "Snippet 2: FKZ DEF456", "Snippet 3: FKZ GHI789"]

        prompt = build_improved_user_prompt(snippets, "Test Query")

        for snippet in snippets:
            assert snippet in prompt

    def test_user_prompt_includes_query(self):
        """Test: User-Prompt enthält die Query."""
        from src.llm.llm_wrapper import build_improved_user_prompt

        query = "Künstliche Intelligenz Projekte Bayern"
        prompt = build_improved_user_prompt(["Snippet"], query)

        assert query in prompt

    def test_user_prompt_structure(self):
        """Test: User-Prompt hat erwartete Struktur."""
        from src.llm.llm_wrapper import build_improved_user_prompt

        prompt = build_improved_user_prompt(["Test"], "Query")

        assert "KONTEXT" in prompt
        assert "NUTZERANFRAGE" in prompt
        assert "ANWEISUNG" in prompt


class TestSavePromptToMd:
    """Tests für die Prompt-Persistierung."""

    def test_save_prompt_creates_file(self):
        """Test: Prompt wird als Markdown-Datei gespeichert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = "Test prompt content"
            result_path = save_prompt_to_md(prompt, folder=tmpdir)

            assert result_path.exists()
            assert result_path.suffix == ".md"

            content = result_path.read_text(encoding="utf-8")
            assert "Test prompt content" in content
            assert "# LLM Prompt" in content

    def test_save_prompt_creates_directory(self):
        """Test: Zielverzeichnis wird erstellt, falls nicht vorhanden."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "prompts"
            prompt = "Test"

            result_path = save_prompt_to_md(prompt, folder=str(nested_dir))

            assert nested_dir.exists()
            assert result_path.exists()

    def test_save_prompt_with_special_characters(self):
        """Test: Umlaute und Sonderzeichen werden korrekt gespeichert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = "Test mit Ümläüten und €-Zeichen"
            result_path = save_prompt_to_md(prompt, folder=tmpdir)

            content = result_path.read_text(encoding="utf-8")
            assert "Ümläüten" in content
            assert "€-Zeichen" in content


class TestEmbedText:
    """Tests für die Embedding-Erzeugung."""

    @patch("src.llm.llm_wrapper.ollama_embed")
    def test_embed_text_with_embeddings_attribute(self, mock_embed):
        """Test: Embedding-Extraktion aus Response-Objekt mit embeddings-Attribut."""
        mock_response = MagicMock()
        mock_response.embeddings = [[1.0, 2.0, 3.0]]
        mock_embed.return_value = mock_response

        result = embed_text("test text")

        assert result == [1.0, 2.0, 3.0]
        mock_embed.assert_called_once_with(model="nomic-embed-text", input="test text")

    @patch("src.llm.llm_wrapper.ollama_embed")
    def test_embed_text_with_dict_response(self, mock_embed):
        """Test: Embedding-Extraktion aus Dictionary-Response."""
        mock_embed.return_value = {"embeddings": [[1.5, 2.5, 3.5]]}

        result = embed_text("test text")

        assert result == [1.5, 2.5, 3.5]

    @patch("src.llm.llm_wrapper.ollama_embed")
    def test_embed_text_with_flat_embedding(self, mock_embed):
        """Test: Embedding-Extraktion aus flacher Liste."""
        mock_embed.return_value = {"embedding": [1.0, 2.0, 3.0]}

        result = embed_text("test text")

        assert result == [1.0, 2.0, 3.0]

    @patch("src.llm.llm_wrapper.ollama_embed")
    def test_embed_text_with_custom_model(self, mock_embed):
        """Test: Custom Model wird korrekt übergeben."""
        mock_embed.return_value = {"embeddings": [[1.0]]}

        embed_text("test", model="custom-model")

        mock_embed.assert_called_once_with(model="custom-model", input="test")

    @patch("src.llm.llm_wrapper.ollama_embed")
    def test_embed_text_raises_on_invalid_format(self, mock_embed):
        """Test: Exception bei ungültigem Embedding-Format."""
        mock_embed.return_value = {"embeddings": "invalid"}

        with pytest.raises(ValueError, match="unerwartetes Format"):
            embed_text("test")

    @patch("src.llm.llm_wrapper.ollama_embed")
    def test_embed_text_raises_on_unexpected_type(self, mock_embed):
        """Test: Exception bei unerwartetem Rückgabetyp."""
        mock_embed.return_value = "invalid_response"

        with pytest.raises(TypeError, match="Unerwarteter Rückgabewert"):
            embed_text("test")


class TestChatSystemQuery:
    """Tests für die Chat-Completion."""

    @patch("src.llm.llm_wrapper.LLMClient")
    @patch("src.llm.llm_wrapper.save_prompt_to_md")
    def test_chat_system_query_success(self, mock_save, mock_client_class):
        """Test: Erfolgreiche Chat-Completion."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = "Test response"
        mock_client_class.return_value = mock_client

        result = chat_system_query("You are helpful", "What is AI?", model="test-model")

        assert result == "Test response"
        mock_client_class.assert_called_once_with(llm="test-model", max_tokens=1024, temperature=0.5)
        mock_save.assert_called_once()

    @patch("src.llm.llm_wrapper.LLMClient")
    @patch("src.llm.llm_wrapper.save_prompt_to_md")
    def test_chat_system_query_with_none_model(self, mock_save, mock_client_class):
        """Test: Chat-Completion mit Default-Model."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = "Response"
        mock_client_class.return_value = mock_client

        chat_system_query("System", "User", model=None)

        mock_client_class.assert_called_once_with(llm=None, max_tokens=1024, temperature=0.5)

    @patch("src.llm.llm_wrapper.LLMClient")
    @patch("src.llm.llm_wrapper.save_prompt_to_md")
    def test_chat_system_query_raises_on_error(self, mock_save, mock_client_class):
        """Test: Exception wird weitergeleitet."""
        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = RuntimeError("API Error")
        mock_client_class.return_value = mock_client

        with pytest.raises(RuntimeError, match="API Error"):
            chat_system_query("System", "User")


class TestPromptGeneration:
    """Tests für die Prompt-Generierung."""

    def test_get_improved_system_prompt_structure(self):
        """Test: System-Prompt enthält erwartete Strukturelemente."""
        prompt = get_improved_system_prompt()

        assert "KI-Assistent" in prompt
        assert "BMBF" in prompt
        assert "FKZ" in prompt
        assert "Aufgaben" in prompt
        assert "Regeln" in prompt

    def test_build_improved_user_prompt_with_snippets(self):
        """Test: User-Prompt wird korrekt mit Snippets formatiert."""
        snippets = ["1. FKZ: ABC123 | Empfänger: Uni XY", "2. FKZ: DEF456 | Empfänger: Institut YZ"]
        query = "KI Projekte"

        prompt = build_improved_user_prompt(snippets, query)

        assert "KONTEXT" in prompt
        assert "ABC123" in prompt
        assert "DEF456" in prompt
        assert "KI Projekte" in prompt
        assert "ANWEISUNG" in prompt

    def test_build_improved_user_prompt_empty_snippets(self):
        """Test: User-Prompt funktioniert mit leerer Snippet-Liste."""
        prompt = build_improved_user_prompt([], "Test query")

        assert "KONTEXT" in prompt
        assert "Test query" in prompt

    def test_build_improved_user_prompt_special_characters(self):
        """Test: Sonderzeichen in Snippets und Query werden korrekt behandelt."""
        snippets = ["Förderung: €1.5M"]
        query = "Was kostet die Förderung?"

        prompt = build_improved_user_prompt(snippets, query)

        assert "€1.5M" in prompt
        assert "Was kostet die Förderung?" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
