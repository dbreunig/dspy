# AGENTS.md

Guidance for coding agents working in this repository.

## Project layout

- `dspy/` — the library.
  - `dspy/signatures/` — `Signature`, `InputField`, `OutputField`; string-signature parsing lives in `signature.py`.
  - `dspy/predict/` — prediction modules (`Predict`, `ChainOfThought`, `ReAct`, ...).
  - `dspy/primitives/` — `Module` base class, `Example`, `Prediction`.
  - `dspy/adapters/` — translate signatures to/from LM message formats (chat, JSON, XML, two-step).
  - `dspy/clients/` — `dspy.LM` and provider integrations (built on LiteLLM), caching, fine-tuning.
  - `dspy/teleprompt/` — optimizers (`BootstrapFewShot`, `MIPROv2`, `GEPA`, ...).
  - `dspy/evaluate/` — `Evaluate` and metrics.
  - `dspy/utils/` — callbacks, exceptions, parallelizer, logging.
  - `dspy/dsp/utils/settings.py` — global settings and `dspy.context`.
- `tests/` — pytest suite, mirroring the `dspy/` layout.
- `docs/` — MkDocs site (https://dspy.ai).

## Running tests and lint

- Install dev deps: `uv sync --extra dev`
- Run tests: `uv run pytest tests/` (or a subset, e.g. `uv run pytest tests/utils/ -q`)
- Lint: `uv run ruff check .` (autofix with `--fix`; formatting via `uv run ruff format`)

Some tests require network access or provider credentials; unrelated failures
in those tests are usually environmental.

## Conventions that matter

- Sync/async twins: modules expose `forward`/`aforward` and are invoked via
  `__call__`/`acall`. When you change one side, keep the other in sync — they
  must stay behaviorally equivalent.
- String signatures (`"question -> answer"`) are parsed at runtime. Prefer
  class-based `dspy.Signature` subclasses in examples and docs; they are
  explicit about types and instructions.
- Threading/async: `dspy.configure(...)` may only be called from the thread
  that first configured settings. Inside worker threads or async tasks, use
  `with dspy.context(...)` to apply overrides locally.
- Exceptions: new exception types should subclass `DSPyError` from
  `dspy/utils/exceptions.py`.
- Callbacks: the `@with_callbacks` decorator (`dspy/utils/callback.py`)
  short-circuits when no callbacks are configured — don't add per-call
  overhead on that fast path.
