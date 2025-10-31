"""Unit-Tests für src/llm/llm_wrapper.py

Tests für:
- Prompt-Persistierung
- Embedding-Erzeugung
- Chat-Completion
- Prompt-Generierung
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.llm.llm_wrapper import (
    save_prompt_to_md,
    chat_system_query,
    embed_text,
    get_improved_system_prompt,
    build_improved_user_prompt
)


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

    @patch('src.llm.llm_wrapper.ollama_embed')
    def test_embed_text_with_embeddings_attribute(self, mock_embed):
        """Test: Embedding-Extraktion aus Response-Objekt mit embeddings-Attribut."""
        mock_response = MagicMock()
        mock_response.embeddings = [[1.0, 2.0, 3.0]]
        mock_embed.return_value = mock_response
        
        result = embed_text("test text")
        
        assert result == [1.0, 2.0, 3.0]
        mock_embed.assert_called_once_with(model="nomic-embed-text", input="test text")

    @patch('src.llm.llm_wrapper.ollama_embed')
    def test_embed_text_with_dict_response(self, mock_embed):
        """Test: Embedding-Extraktion aus Dictionary-Response."""
        mock_embed.return_value = {"embeddings": [[1.5, 2.5, 3.5]]}
        
        result = embed_text("test text")
        
        assert result == [1.5, 2.5, 3.5]

    @patch('src.llm.llm_wrapper.ollama_embed')
    def test_embed_text_with_flat_embedding(self, mock_embed):
        """Test: Embedding-Extraktion aus flacher Liste."""
        mock_embed.return_value = {"embedding": [1.0, 2.0, 3.0]}
        
        result = embed_text("test text")
        
        assert result == [1.0, 2.0, 3.0]

    @patch('src.llm.llm_wrapper.ollama_embed')
    def test_embed_text_with_custom_model(self, mock_embed):
        """Test: Custom Model wird korrekt übergeben."""
        mock_embed.return_value = {"embeddings": [[1.0]]}
        
        embed_text("test", model="custom-model")
        
        mock_embed.assert_called_once_with(model="custom-model", input="test")

    @patch('src.llm.llm_wrapper.ollama_embed')
    def test_embed_text_raises_on_invalid_format(self, mock_embed):
        """Test: Exception bei ungültigem Embedding-Format."""
        mock_embed.return_value = {"embeddings": "invalid"}
        
        with pytest.raises(ValueError, match="unerwartetes Format"):
            embed_text("test")

    @patch('src.llm.llm_wrapper.ollama_embed')
    def test_embed_text_raises_on_unexpected_type(self, mock_embed):
        """Test: Exception bei unerwartetem Rückgabetyp."""
        mock_embed.return_value = "invalid_response"
        
        with pytest.raises(TypeError, match="Unerwarteter Rückgabewert"):
            embed_text("test")


class TestChatSystemQuery:
    """Tests für die Chat-Completion."""

    @patch('src.llm.llm_wrapper.LLMClient')
    @patch('src.llm.llm_wrapper.save_prompt_to_md')
    def test_chat_system_query_success(self, mock_save, mock_client_class):
        """Test: Erfolgreiche Chat-Completion."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = "Test response"
        mock_client_class.return_value = mock_client
        
        result = chat_system_query(
            "You are helpful",
            "What is AI?",
            model="test-model"
        )
        
        assert result == "Test response"
        mock_client_class.assert_called_once_with(llm="test-model")
        mock_save.assert_called_once()

    @patch('src.llm.llm_wrapper.LLMClient')
    @patch('src.llm.llm_wrapper.save_prompt_to_md')
    def test_chat_system_query_with_none_model(self, mock_save, mock_client_class):
        """Test: Chat-Completion mit Default-Model."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = "Response"
        mock_client_class.return_value = mock_client
        
        chat_system_query("System", "User", model=None)
        
        mock_client_class.assert_called_once_with(llm=None)

    @patch('src.llm.llm_wrapper.LLMClient')
    @patch('src.llm.llm_wrapper.save_prompt_to_md')
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
        snippets = [
            "1. FKZ: ABC123 | Empfänger: Uni XY",
            "2. FKZ: DEF456 | Empfänger: Institut YZ"
        ]
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
