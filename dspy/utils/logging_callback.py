"""Console tracer that prints a live, indented span tree of a DSPy program's execution."""

from __future__ import annotations

import sys
import time
from typing import Any, TextIO

from dspy.utils.callback import BaseCallback
from dspy.utils.callback_context import ACTIVE_CALL_ID


class LoggingCallback(BaseCallback):
    """A callback that prints a live, indented span tree of a program's execution.

    Each module, LM, adapter, and tool call is printed as it starts (``▸``) and again as it
    finishes (``✓``, or ``✗`` with the exception on failure), with two spaces of indentation
    per nesting level and the elapsed wall-clock time of each span.

    Example:

    ```
    import dspy
    from dspy.utils import LoggingCallback

    dspy.configure(callbacks=[LoggingCallback()])

    cot = dspy.ChainOfThought("question -> answer")
    cot(question="What is the meaning of life?")

    # > ▸ ChainOfThought
    # >   ▸ Predict(question -> reasoning, answer)
    # >     ▸ ChatAdapter.format
    # >     ✓ ChatAdapter.format (0.00s)
    # >     ▸ LM(openai/gpt-4o-mini)
    # >     ✓ LM(openai/gpt-4o-mini) (0.72s)
    # >     ▸ ChatAdapter.parse
    # >     ✓ ChatAdapter.parse (0.00s)
    # >   ✓ Predict(question -> reasoning, answer) (0.73s)
    # > ✓ ChainOfThought (0.73s)
    ```

    Args:
        file: An optional file-like object to write output to (must have a `.write()`
            method). When provided, ANSI color codes are automatically disabled.
            Defaults to `None` (prints to stdout).
        color: Whether to colorize the ``▸``/``✓``/``✗`` markers with ANSI codes.
            Ignored (treated as `False`) when `file` is provided.
        verbose: Whether to include adapter ``format``/``parse`` spans in the tree.
            When `False`, only module, LM, tool, and evaluate spans are printed.
    """

    def __init__(self, file: TextIO | None = None, color: bool = True, verbose: bool = True):
        self.file = file
        self.color = color and file is None
        self.verbose = verbose
        # call_id -> (depth, label, start time from time.perf_counter())
        self._calls: dict[str, tuple[int, str, float]] = {}

    def _write(self, line: str) -> None:
        out = self.file or sys.stdout
        print(line, file=out, flush=True)

    def _colorize(self, text: str, ansi_code: str) -> str:
        if self.color:
            return f"\x1b[{ansi_code}m{text}\x1b[0m"
        return text

    def _on_start(self, call_id: str, label: str, print_line: bool = True) -> None:
        # `with_callbacks` fires start handlers before setting ACTIVE_CALL_ID to the new
        # call_id, so at this point ACTIVE_CALL_ID still holds the parent call's id.
        parent_call_id = ACTIVE_CALL_ID.get()
        parent = self._calls.get(parent_call_id) if parent_call_id is not None else None
        depth = parent[0] + 1 if parent is not None else 0
        self._calls[call_id] = (depth, label, time.perf_counter())
        if print_line:
            self._write(f"{'  ' * depth}{self._colorize('▸', '34')} {label}")

    def _on_end(self, call_id: str, fallback_label: str, exception: Exception | None) -> None:
        record = self._calls.pop(call_id, None)
        if record is None:
            # No recorded start (e.g. the callback was attached mid-run); print at depth 0
            # without elapsed time rather than crashing.
            depth, label, elapsed = 0, fallback_label, None
        else:
            depth, label, start = record
            elapsed = time.perf_counter() - start

        marker = self._colorize("✗", "31") if exception is not None else self._colorize("✓", "32")
        line = f"{'  ' * depth}{marker} {label}"
        if elapsed is not None:
            line += f" ({elapsed:.2f}s)"
        if exception is not None:
            line += f" - {type(exception).__name__}: {exception}"
        self._write(line)

    def _module_label(self, instance: Any) -> str:
        name = type(instance).__name__
        # Predict-like modules carry a signature; render it when available.
        signature = getattr(instance, "signature", None)
        signature_str = getattr(signature, "signature", None)
        if isinstance(signature_str, str):
            return f"{name}({signature_str})"
        return name

    def _lm_label(self, instance: Any) -> str:
        model = getattr(instance, "model", None)
        if isinstance(model, str):
            return f"LM({model})"
        return type(instance).__name__

    def _tool_label(self, instance: Any) -> str:
        tool_name = getattr(instance, "name", None)
        if isinstance(tool_name, str):
            return f"Tool({tool_name})"
        return type(instance).__name__

    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self._on_start(call_id, self._module_label(instance))

    def on_module_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None):
        self._on_end(call_id, "Module", exception)

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self._on_start(call_id, self._lm_label(instance))

    def on_lm_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        self._on_end(call_id, "LM", exception)

    def on_adapter_format_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self._on_start(call_id, f"{type(instance).__name__}.format", print_line=self.verbose)

    def on_adapter_format_end(
        self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None
    ):
        if not self.verbose:
            self._calls.pop(call_id, None)
            return
        self._on_end(call_id, "Adapter.format", exception)

    def on_adapter_parse_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self._on_start(call_id, f"{type(instance).__name__}.parse", print_line=self.verbose)

    def on_adapter_parse_end(
        self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None
    ):
        if not self.verbose:
            self._calls.pop(call_id, None)
            return
        self._on_end(call_id, "Adapter.parse", exception)

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self._on_start(call_id, self._tool_label(instance))

    def on_tool_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        self._on_end(call_id, "Tool", exception)

    def on_evaluate_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self._on_start(call_id, type(instance).__name__)

    def on_evaluate_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None):
        self._on_end(call_id, "Evaluate", exception)
