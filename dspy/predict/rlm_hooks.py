"""
Review hooks for RLM.

RLM can call a ``before_execute`` hook after the model proposes code and an
``after_execute`` hook after the interpreter runs it. Each hook receives a step
and returns a decision, or ``None`` to let the step proceed unchanged.

Decisions a ``before_execute`` hook may return:
- ``Run(code=None)``: run the proposed code, or run ``code`` in its place.
- ``Reject(feedback)``: skip execution and show ``feedback`` to the model.
- ``Finish(**outputs)``: end the run with these outputs.

Decisions an ``after_execute`` hook may return:
- ``Reject(feedback)``: show ``feedback`` to the model beside the output. A rejected SUBMIT does not end the run.
- ``Replace(output)``: show ``output`` to the model in place of the real output. The run continues.
- ``Finish(**outputs)``: end the run with these outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from dspy.primitives.code_interpreter import CodeInterpreter
    from dspy.primitives.repl_types import REPLHistory

__all__ = [
    "ProposedStep",
    "ExecutedStep",
    "Run",
    "Reject",
    "Replace",
    "Finish",
    "BeforeExecuteHook",
    "AfterExecuteHook",
]


@dataclass(frozen=True)
class ProposedStep:
    """Code the model proposed, before it runs.

    Attributes:
        iteration: Zero-based index of this iteration.
        reasoning: The model's reasoning for this step.
        code: The proposed code, with markdown fences stripped.
        history: The REPL history before this step.
        repl: The live interpreter. A hook may run its own code here; that code shares the model's namespace.
    """

    iteration: int
    reasoning: str
    code: str
    history: REPLHistory
    repl: CodeInterpreter


@dataclass(frozen=True)
class ExecutedStep(ProposedStep):
    """A step after the interpreter ran it.

    Attributes:
        code: The code that ran, which differs from the proposal if ``before_execute`` edited it.
        result: The raw interpreter result: printed output, a ``FinalOutput``, or an ``"[Error] ..."`` string.
        output: The text the model will see for this step.
        final_outputs: The parsed outputs if the code called SUBMIT with valid outputs, else ``None``.
    """

    result: Any
    output: str
    final_outputs: dict[str, Any] | None


@dataclass(frozen=True)
class Run:
    """Run the step's code, or run ``code`` in its place."""

    code: str | None = None


@dataclass(frozen=True)
class Reject:
    """Reject the step and show ``feedback`` to the model."""

    feedback: str


@dataclass(frozen=True)
class Replace:
    """Show ``output`` to the model in place of the real output."""

    output: str


class Finish:
    """End the run with the given output fields."""

    def __init__(self, **outputs: Any):
        self.outputs = outputs

    def __repr__(self) -> str:
        return f"Finish({self.outputs!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Finish) and self.outputs == other.outputs


BeforeDecision = Run | Reject | Finish | None
AfterDecision = Reject | Replace | Finish | None
BeforeExecuteHook = Callable[[ProposedStep], BeforeDecision | Awaitable[BeforeDecision]]
AfterExecuteHook = Callable[[ExecutedStep], AfterDecision | Awaitable[AfterDecision]]

BEFORE_DECISIONS = (Run, Reject, Finish)
AFTER_DECISIONS = (Reject, Replace, Finish)
