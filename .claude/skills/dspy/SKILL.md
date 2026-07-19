---
name: dspy
description: >-
  Write, evaluate, and optimize DSPy programs — the Python framework for
  programming (not prompting) language models with signatures, modules, and
  optimizers. Use this skill whenever the user mentions DSPy, dspy.Module,
  signatures, ChainOfThought, ReAct, MIPROv2, GEPA, BootstrapFewShot,
  teleprompters, or "prompt optimization"; whenever a repo imports dspy; and
  whenever the user wants to build, debug, evaluate, or optimize an LLM
  pipeline, agent, RAG system, classifier, or extractor and DSPy is available
  or requested — even if they don't say "DSPy" but are working in a DSPy
  codebase.
---

# DSPy

DSPy is a framework for *programming* language models instead of prompting them.
You declare what a step does (a **Signature**: typed inputs → outputs), choose a
strategy for executing it (a **Module**: Predict, ChainOfThought, ReAct, ...),
and compose modules in plain Python. Prompt construction and output parsing are
handled by **Adapters**, and prompts/demos are improved automatically by
**Optimizers** against a metric. This separation is the whole point: never
hand-tune prompt wording that an optimizer could learn, and never bake
formatting into strings that an adapter should own.

The intended workflow has three stages — respect the order:

1. **Programming**: define the task, write signatures and modules, get it running.
2. **Evaluation**: collect a dev set, write a metric, measure with `dspy.Evaluate`.
3. **Optimization**: compile with an optimizer (MIPROv2, GEPA, ...) against that metric.

It is unproductive to optimize a poorly designed program or a bad metric, so
don't reach for an optimizer before stages 1–2 exist.

Reference files in this skill (read them when the task touches their area):

- `references/patterns.md` — task-type → pattern cookbook with canonical code:
  RAG, agents/tools/MCP, multi-hop, classification, extraction, multi-stage
  pipelines, conversation history, multimodal, BestOfN/Refine, PoT/RLM.
- `references/optimization.md` — datasets, metrics, `dspy.Evaluate`, the full
  optimizer catalog with decision guidance, GEPA specifics, finetuning.
- `references/production.md` — saving/loading, caching, async, streaming,
  deployment, observability/debugging, parallelism, adapters, error types.

## Setup

```python
import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))   # LiteLLM "provider/model" string
```

- Model strings: `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4-5-20250929`,
  `gemini/gemini-2.5-pro`, `ollama_chat/llama3.2` (with `api_base=`,
  `api_key=''`), any OpenAI-compatible server via
  `dspy.LM('openai/model-name', api_base=..., api_key=...)`.
- Generation params live on the LM, not in settings:
  `dspy.LM('openai/gpt-4o-mini', temperature=0.9, max_tokens=3000, cache=False)`.
- For OpenAI reasoning models use `model_type="responses"` and generous
  `max_tokens` (e.g. 16000) with `temperature=1.0`.
- `dspy.configure(...)` is process-wide and may only be called again from the
  thread that first called it. Inside threads, async tasks, or for scoped
  overrides (e.g. a bigger judge model for one step), use the context manager:

```python
with dspy.context(lm=dspy.LM("openai/gpt-4o")):
    verdict = self.judge(...)
```

`dspy.context` propagates across `await`, but NOT into bare `threading.Thread`
or `ThreadPoolExecutor` workers — use `dspy.Parallel`, `module.batch`, or
`dspy.asyncify`, which snapshot and reapply settings.

## Signatures

A signature declares typed fields, not prompt text. Inline form for quick
cases, class form when you want instructions or field descriptions:

```python
# Inline: types default to str; annotate when they don't
dspy.Predict("question -> answer")
dspy.ChainOfThought("context: list[str], question: str -> answer: str")
dspy.Predict("sentence -> sentiment: bool")

# Class-based: the docstring IS the task instruction
class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""
    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField()
    faithfulness: bool = dspy.OutputField()
    evidence: dict[str, list[str]] = dspy.OutputField(desc="supporting evidence for claims")
```

Field types can be anything Pydantic can validate: `str`, `int`, `bool`,
`float`, `list[...]`, `dict[str, Any]`, `Optional[...]`, `Literal[...]`, Enums,
Pydantic models, and DSPy media types (`dspy.Image`, `dspy.Audio`,
`dspy.History`, `dspy.Tool`, `dspy.ToolCalls`, `dspy.Code`). For
classification, a `Literal[...]` or `str`-Enum output field is the canonical
pattern.

Two rules that matter more than they look:

1. **Field names are load-bearing.** The LM reads them; `"a, b -> c"` produces
   garbage while `"location, mood -> haiku"` works. Names are also your
   program's API (`result.haiku`) and optimizers never rename fields or rewrite
   `desc` — they only rewrite the instruction docstring. A bad field name
   cannot be fixed by optimization, so choose names carefully up front.
2. **Don't over-write instructions.** State the task fundamentals; resist
   restating what the signature already says or writing prescriptive rule
   lists. Expansive guidance, watch-outs, and phrasing tweaks are exactly what
   optimizers are for.

Inline signature with instructions, when a class feels heavy:

```python
sig = dspy.Signature("claim -> titles: list[str]",
                     "Find all Wikipedia titles relevant to verifying the claim.")
```

## Built-in modules

Declare with a signature, call with kwargs, read typed attributes off the
returned `dspy.Prediction`:

```python
classify = dspy.ChainOfThought("sentence -> sentiment: bool")
pred = classify(sentence="it works!")   # pred.sentiment, pred.reasoning
```

| Module | Use for |
|---|---|
| `dspy.Predict(sig)` | The base predictor; simple, well-specified steps |
| `dspy.ChainOfThought(sig)` | Adds a `reasoning` output field; often a free quality win over Predict |
| `dspy.ReAct(sig, tools=[...], max_iters=20)` | Agents. Tools are plain Python functions with type hints + docstrings |
| `dspy.ProgramOfThought(sig)` | Answers by generating and executing Python in a sandbox (math, computation) |
| `dspy.CodeAct(sig, tools=[...])` | ReAct that acts by writing code; tools must be pure top-level `def`s |
| `dspy.BestOfN(module, N, reward_fn, threshold)` | Sample N times, keep the best per `reward_fn(args, pred) -> float` |
| `dspy.Refine(module, N, reward_fn, threshold)` | BestOfN plus LM-generated feedback between attempts |
| `dspy.RLM(sig, max_iters=20)` | Experimental: REPL-driving agent for huge contexts / DataFrames |
| `dspy.MultiChainComparison(sig, M=3)` | Compare M completions and synthesize one answer |

`dspy.Assert` / `dspy.Suggest` are **deprecated and removed** — use
`dspy.Refine` or `dspy.BestOfN` for output constraints.

ReAct essentials: pass functions directly (`dspy.ReAct("question -> answer",
tools=[search, calculate])`); each tool needs type hints and a docstring; tool
exceptions become observations the agent can react to, not crashes; results
carry `pred.trajectory` (`thought_i` / `tool_name_i` / `tool_args_i` /
`observation_i`). MCP tools convert via `dspy.Tool.from_mcp_tool(session, tool)`
(then use `await agent.acall(...)` — MCP tools are async); LangChain tools via
`dspy.Tool.from_langchain(tool)`.

## Custom modules

Compose steps in a `dspy.Module` subclass: sub-modules as attributes in
`__init__` (assignment is registration), logic in `forward`, return a
`dspy.Prediction`. This is the recommended home for any multi-step logic — it
is what makes optimizers, saving, tracing, and streaming see your program.

```python
class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        context = search(question).passages     # any Python: retrievers, APIs, DBs
        return self.respond(context=context, question=question)

rag = RAG()
result = rag(question="...")    # call the instance — NEVER rag.forward(...)
```

- Attribute names become predictor names (`named_predictors()`) used by
  optimizers, saving, and streaming — name them meaningfully
  (`self.generate_query`, not `self.p1`).
- Calling `.forward()` directly bypasses callbacks, usage tracking, and tracing
  (and logs a deprecation warning). Always invoke the instance.
- Loops, branching, try/except, and non-LM side effects inside `forward` are
  all fine — see the multi-hop and agent patterns in `references/patterns.md`.
- For async, implement `aforward` and invoke with `await module.acall(...)`.
- Sub-modules hidden in closures or custom containers (anything other than
  attributes, lists, dicts) are invisible to optimizers and `save()`.

## Evaluate, then optimize

The 60-second version — read `references/optimization.md` before actually
running an optimizer:

```python
# 1. Data: Examples with declared inputs (everything else is a label)
trainset = [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in data]

# 2. Metric: (example, pred, trace=None) -> bool | float
def metric(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# 3. Evaluate
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=16,
                         display_progress=True, display_table=5)
result = evaluate(program)          # result.score, result.results

# 4. Optimize (compile returns a new, improved program)
optimizer = dspy.MIPROv2(metric=metric, auto="light", num_threads=16)
optimized = optimizer.compile(program, trainset=trainset)
optimized.save("program.json")      # human-readable state; inspect what it learned
```

Optimizer cheat-sheet (details and exact signatures in the reference):
~10 examples → `BootstrapFewShot`; 50+ → `BootstrapFewShotWithRandomSearch`;
200+ and a real budget → `MIPROv2`; instruction-only / 0-shot → MIPROv2 with
`max_bootstrapped_demos=0, max_labeled_demos=0`; feedback-rich reflective
optimization → `GEPA` (needs a strong `reflection_lm` and a 5-arg metric that
returns `dspy.Prediction(score=..., feedback=...)`); distill into a small model
→ `BootstrapFinetune`.

The `trace` convention: metrics receive `trace=None` during evaluation (return
a graded float) and a non-None trace during bootstrapping (return a strict
bool) — this is how optimizers decide which demos are good enough to keep:

```python
def metric(example, pred, trace=None):
    score = compute_partial_credit(example, pred)
    if trace is not None:
        return score >= 1.0     # strict when selecting demos
    return score
```

Loading a saved program requires reconstructing the same architecture first:
`fresh = RAG(); fresh.load("program.json")`. Whole-program saving
(`save_program=True`, a directory, loaded via `dspy.load(dir)`) avoids that but
uses pickle — trusted files only.

## Debugging

- `dspy.inspect_history(n=1)` — print the last n actual LM calls (the real
  prompt the adapter built, and the raw response). First stop for "why did it
  output that?".
- `dspy.configure(track_usage=True)` then `pred.get_lm_usage()` for tokens;
  `sum(x['cost'] for x in lm.history if x['cost'])` for spend.
- Caching is ON by default (memory + disk). Identical calls return cached
  results — great for iteration, confusing when you expect fresh samples. For
  a fresh sample keep the cache but vary `rollout_id` (requires nonzero
  temperature): `predict(..., config={"rollout_id": 1, "temperature": 1.0})`.
  Disable outright with `dspy.configure_cache(enable_disk_cache=False,
  enable_memory_cache=False)` or per-LM `cache=False`.
- Structured exceptions: `dspy.LMError` and subclasses
  (`ContextWindowExceededError`, `LMRateLimitError`, ...) with `.model`,
  `.provider`, `.retry_after`.
- MLflow tracing (`mlflow.dspy.autolog()`) gives per-step traces including
  tools and retrievers — see `references/production.md`.

## Gotchas worth knowing before they bite

1. Cached identical calls mean "it keeps giving the same answer" is usually the
   cache, not the model. `rollout_id` + `temperature > 0` for diversity.
2. `dspy.configure` from a non-owner thread raises `RuntimeError`; use
   `dspy.context`. Plain threads don't inherit `dspy.context` overrides.
3. Optimizers rewrite only instruction docstrings — field names and `desc` are
   inert, so get names right first; and don't hand-tune instruction wording
   the optimizer will replace anyway.
4. `Evaluate` swallows per-example failures into `failure_score=0.0` (up to
   `max_errors`) — a silently low score may be crashes, not bad answers. Set
   `provide_traceback=True` when scores look wrong.
5. ChatAdapter (default) silently falls back to JSONAdapter on parse failures.
   In tests, `dspy.ChatAdapter(use_json_adapter_fallback=False)` to surface
   real errors. For models with native structured output, `dspy.configure(
   adapter=dspy.JSONAdapter())` is leaner.
6. Marking label fields as inputs in `.with_inputs(...)` leaks answers into
   prompts. Inputs = what the program sees; everything else is for the metric.
7. Old API traps: `dspy.Assert`/`dspy.Suggest` are removed (use
   Refine/BestOfN); `Evaluate(return_outputs=...)` is removed (use
   `result.results`); typed predictors (`TypedPredictor`) are gone — types on
   ordinary signatures do that job now.
8. State-only `load()` needs the exact same program class/signature to be
   reconstructed first; version mismatches between save/load warn but can
   silently degrade — pin DSPy versions across save/load.
