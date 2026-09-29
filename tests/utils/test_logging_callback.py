import io
import re

import pytest

import dspy
from dspy.utils.dummies import DummyLM
from dspy.utils.logging_callback import LoggingCallback

ELAPSED_PATTERN = re.compile(r"\(\d+\.\d{2}s\)")


def test_predict_produces_indented_span_tree():
    buf = io.StringIO()
    lm = DummyLM([{"answer": "Paris"}])

    with dspy.context(lm=lm, callbacks=[LoggingCallback(file=buf)]):
        dspy.Predict("question -> answer")(question="What is the capital of France?")

    lines = buf.getvalue().splitlines()

    # Module at depth 0, adapter format/parse and LM at depth 1.
    assert lines[0] == "▸ Predict(question -> answer)"
    assert "  ▸ ChatAdapter.format" in lines
    assert "  ▸ LM(dummy)" in lines
    assert "  ▸ ChatAdapter.parse" in lines
    assert any(line.startswith("  ✓ ChatAdapter.format (") for line in lines)
    assert any(line.startswith("  ✓ LM(dummy) (") for line in lines)
    assert any(line.startswith("  ✓ ChatAdapter.parse (") for line in lines)
    assert lines[-1].startswith("✓ Predict(question -> answer) (")

    # The LM span starts after the module span and ends before it.
    assert lines.index("  ▸ LM(dummy)") > lines.index("▸ Predict(question -> answer)")


def test_nested_module_adds_indentation_level():
    class Wrapper(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict("question -> answer")

        def forward(self, question):
            return self.predict(question=question)

    buf = io.StringIO()
    lm = DummyLM([{"answer": "Paris"}])

    with dspy.context(lm=lm, callbacks=[LoggingCallback(file=buf)]):
        Wrapper()(question="What is the capital of France?")

    lines = buf.getvalue().splitlines()

    assert lines[0] == "▸ Wrapper"
    assert lines[1] == "  ▸ Predict(question -> answer)"
    assert "    ▸ LM(dummy)" in lines
    assert any(line.startswith("  ✓ Predict(question -> answer) (") for line in lines)
    assert lines[-1].startswith("✓ Wrapper (")


def test_failing_tool_prints_error_line():
    def search(query: str) -> str:
        """A tool that always fails."""
        raise ValueError("boom")

    tool = dspy.Tool(search)
    buf = io.StringIO()

    with dspy.context(callbacks=[LoggingCallback(file=buf)]):
        with pytest.raises(ValueError, match="boom"):
            tool(query="anything")

    lines = buf.getvalue().splitlines()

    assert lines[0] == "▸ Tool(search)"
    assert lines[1].startswith("✗ Tool(search) (")
    assert "ValueError: boom" in lines[1]


def test_failing_module_prints_error_line():
    class Exploder(dspy.Module):
        def forward(self):
            raise RuntimeError("nope")

    buf = io.StringIO()

    with dspy.context(callbacks=[LoggingCallback(file=buf)]):
        with pytest.raises(RuntimeError, match="nope"):
            Exploder()()

    lines = buf.getvalue().splitlines()

    assert lines[0] == "▸ Exploder"
    assert lines[1].startswith("✗ Exploder (")
    assert "RuntimeError: nope" in lines[1]


def test_file_output_has_no_ansi_codes_and_elapsed_format():
    buf = io.StringIO()
    lm = DummyLM([{"answer": "Paris"}])

    # color=True is auto-disabled because a file is provided.
    with dspy.context(lm=lm, callbacks=[LoggingCallback(file=buf, color=True)]):
        dspy.Predict("question -> answer")(question="What is the capital of France?")

    output = buf.getvalue()

    assert "\x1b" not in output
    # Every end line carries an elapsed time formatted as (N.NNs).
    end_lines = [line for line in output.splitlines() if line.lstrip().startswith("✓")]
    assert end_lines
    for line in end_lines:
        assert ELAPSED_PATTERN.search(line), line


def test_stdout_output_uses_ansi_colors(capsys):
    lm = DummyLM([{"answer": "Paris"}])

    with dspy.context(lm=lm, callbacks=[LoggingCallback()]):
        dspy.Predict("question -> answer")(question="What is the capital of France?")

    output = capsys.readouterr().out

    assert "\x1b[34m▸\x1b[0m Predict(question -> answer)" in output
    assert "\x1b[32m✓\x1b[0m" in output


def test_verbose_false_hides_adapter_spans():
    buf = io.StringIO()
    lm = DummyLM([{"answer": "Paris"}])

    with dspy.context(lm=lm, callbacks=[LoggingCallback(file=buf, verbose=False)]):
        dspy.Predict("question -> answer")(question="What is the capital of France?")

    lines = buf.getvalue().splitlines()

    assert not any("ChatAdapter" in line for line in lines)
    assert lines[0] == "▸ Predict(question -> answer)"
    assert "  ▸ LM(dummy)" in lines
    assert any(line.startswith("  ✓ LM(dummy) (") for line in lines)
    assert lines[-1].startswith("✓ Predict(question -> answer) (")


def test_verbose_false_unmatched_adapter_end_is_silent():
    buf = io.StringIO()
    callback = LoggingCallback(file=buf, verbose=False)

    callback.on_adapter_format_end(call_id="never-started", outputs=None, exception=None)
    callback.on_adapter_parse_end(call_id="also-never-started", outputs=None, exception=None)

    assert buf.getvalue() == ""


def test_unmatched_end_event_does_not_raise():
    buf = io.StringIO()
    callback = LoggingCallback(file=buf)

    callback.on_module_end(call_id="never-started", outputs=None, exception=None)
    callback.on_lm_end(call_id="also-never-started", outputs=None, exception=ValueError("late"))

    lines = buf.getvalue().splitlines()

    # Printed at depth 0 with the fallback label and no elapsed time.
    assert lines[0] == "✓ Module"
    assert lines[1] == "✗ LM - ValueError: late"
