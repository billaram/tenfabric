"""Tests for dataset formatting."""

from __future__ import annotations

from tenfabric.training.data import _format_alpaca, _format_sharegpt


class TestFormatAlpaca:
    def test_with_input(self):
        example = {
            "instruction": "Translate to French",
            "input": "Hello world",
            "output": "Bonjour le monde",
        }
        result = _format_alpaca(example)
        assert "text" in result
        assert "Translate to French" in result["text"]
        assert "Hello world" in result["text"]
        assert "Bonjour le monde" in result["text"]
        assert "### Input:" in result["text"]

    def test_without_input(self):
        example = {
            "instruction": "Tell me a joke",
            "input": "",
            "output": "Why did the chicken cross the road?",
        }
        result = _format_alpaca(example)
        assert "### Input:" not in result["text"]
        assert "Tell me a joke" in result["text"]
        assert "Why did the chicken" in result["text"]

    def test_missing_fields(self):
        example = {"instruction": "Hello"}
        result = _format_alpaca(example)
        assert "Hello" in result["text"]

    def test_empty_example(self):
        result = _format_alpaca({})
        assert "text" in result


class TestFormatShareGPT:
    def test_basic_conversation(self):
        example = {
            "conversations": [
                {"from": "human", "value": "What is Python?"},
                {"from": "gpt", "value": "Python is a programming language."},
            ]
        }
        result = _format_sharegpt(example)
        assert "### Human:" in result["text"]
        assert "What is Python?" in result["text"]
        assert "### Assistant:" in result["text"]
        assert "programming language" in result["text"]

    def test_with_system_message(self):
        example = {
            "conversations": [
                {"from": "system", "value": "You are helpful."},
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        result = _format_sharegpt(example)
        assert "### System:" in result["text"]
        assert "You are helpful" in result["text"]

    def test_role_key_variants(self):
        """Should handle both 'from'/'value' and 'role'/'content' keys."""
        example = {
            "conversations": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        result = _format_sharegpt(example)
        assert "### Human:" in result["text"]
        assert "### Assistant:" in result["text"]

    def test_empty_conversations(self):
        result = _format_sharegpt({"conversations": []})
        assert result["text"] == ""

    def test_missing_conversations_key(self):
        result = _format_sharegpt({})
        assert result["text"] == ""
