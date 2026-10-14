import pytest

from scripts.qwen36_heretic.prepare_calibration import retokenize


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, **kwargs):
        return {"input_ids": [1] * int(text)}


def test_explicit_calibration_limits_and_accounting():
    entries = [{"messages": [{"content": str(n)}]} for n in (8, 65, 5000)]
    rows, settings = retokenize(entries, Tokenizer(), expected_count=2)
    assert [len(row["input_ids"]) for row in rows] == [65, 4096]
    assert settings["truncated_rows"] == settings["skipped_short_rows"] == 1
    assert settings["fallback_rendered_text"] == 0


def test_chat_failure_cannot_use_stale_rendered_text():
    class Broken(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            raise RuntimeError("wrong template")

    with pytest.raises(ValueError, match="chat template failed"):
        retokenize([{"messages": [{}], "rendered_text": "old"}], Broken())


@pytest.mark.parametrize("entry", [{}, {"messages": []}, {"messages": "invalid"}])
def test_invalid_messages_rejected(entry):
    with pytest.raises(ValueError, match="messages required"):
        retokenize([entry], Tokenizer())


def test_required_row_count():
    with pytest.raises(ValueError, match="Expected 512"):
        retokenize([{"messages": [{"content": "65"}]}], Tokenizer())


def test_boolean_token_ids_rejected():
    class Invalid(Tokenizer):
        def __call__(self, *args, **kwargs):
            return {"input_ids": [True] * 65}

    with pytest.raises(ValueError, match="invalid token IDs"):
        retokenize([{"messages": [{"content": "65"}]}], Invalid())
