# DSPy evaluation & optimization

The optimization loop only pays off when the program design and the metric are
already sound. Work in this order: data → metric → `dspy.Evaluate` baseline →
optimizer. A typical simple run costs on the order of $2 and ~10 minutes;
MIPROv2/GEPA at heavy settings can spend far more, so start with `auto="light"`.

## Datasets

`dspy.Example` is a dict with dot access. `.with_inputs(...)` declares which
fields the program receives; everything else is a label held for the metric:

```python
example = dspy.Example(question="...", answer="...").with_inputs("question")
example.inputs()   # what the program sees
example.labels()   # what the metric sees
```

Careless `.with_inputs` is the classic leak: marking a label as an input puts
the answer in the prompt.

Standard loading pattern (HF/JSON → shuffle with seeded RNG → slice):

```python
examples = [dspy.Example(**row).with_inputs("question") for row in rows]
random.Random(0).shuffle(examples)
trainset, devset, testset = examples[:200], examples[200:500], examples[500:]
```

`dspy.datasets.DataLoader` offers `from_csv/from_json/from_parquet/from_pandas/
from_huggingface(..., fields=..., input_keys=...)`.

Sizes: ~20 examples is already useful for a dev set, 200 goes a long way;
optimizers get real value from 30, aim for 300+. You rarely need labels for
intermediate steps — final outputs (or even just inputs) usually suffice.

**Split guidance (unusual, matters):** for most prompt optimizers, allocate
about **20% train / 80% validation** — they overfit small trainsets, so a
stable valset matters more. **GEPA is the exception**: standard ML convention,
maximize trainset, keep valset just representative. Also:
`BootstrapFewShotWithRandomSearch` silently validates on the trainset if you
don't pass `valset` — pass one.

## Metrics

Signature: `def metric(example, pred, trace=None) -> bool | float`.

The `trace` argument encodes which mode you're in:

- `trace is None` → evaluation: return a graded float.
- `trace` is set → an optimizer is deciding whether a bootstrapped
  demonstration is good enough to keep: return a strict bool.

```python
def metric(example, pred, trace=None):
    answer_match = example.answer.lower() == pred.answer.lower()
    context_match = any(pred.answer.lower() in c for c in pred.context)
    if trace is not None:
        return answer_match and context_match   # strict for demo selection
    return (answer_match + context_match) / 2.0
```

Non-deterministic metrics corrupt demo selection — keep metrics deterministic
during bootstrapping.

**LLM-as-judge**: a judge is just another DSPy program used inside the metric:

```python
class Assess(dspy.Signature):
    """Assess the quality of a text along the specified dimension."""
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: bool = dspy.OutputField()

def metric(gold, pred, trace=None):
    correct = dspy.Predict(Assess)(assessed_text=pred.output,
        assessment_question=f"Does the text answer `{gold.question}` with `{gold.answer}`?")
    return correct.assessment_answer
```

Run judges on a stronger LM via `judge.set_lm(big_lm)` or `dspy.context`. For
pairwise judgments, query in both A/B orders and reconcile (position bias).
Because a metric can itself be a DSPy program, you can optimize the metric too.

Built-ins: `dspy.evaluate.answer_exact_match`, `answer_passage_match`,
`dspy.SemanticF1(decompositional=True)`, `dspy.CompleteAndGrounded()` — the
latter two are modules expecting `example.question/response` and
`pred.response`, usable directly as metrics.

## dspy.Evaluate

```python
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24,
                         display_progress=True, display_table=5,
                         max_errors=100, provide_traceback=True)
result = evaluate(program)
result.score      # float percentage, e.g. 61.0
result.results    # list of (example, prediction, score)
```

Failures are swallowed into `failure_score=0.0` up to `max_errors` — a
mysteriously low score is often crashes; check with `provide_traceback=True`.
(`return_outputs`/`return_all_scores` are removed; use `.results`.)

## Optimizer catalog

All optimizers expose `optimizer.compile(program, trainset=..., [valset=...])`
and return a new, improved program. Save results and inspect what was learned:
`optimized.save("x.json")` (readable JSON);
`optimized.predict.signature.instructions` or loop `named_predictors()`.

**Few-shot family**

- `LabeledFewShot(k=16)` — sticks k labeled examples in the prompt. No metric
  needed; cheapest baseline.
- `BootstrapFewShot(metric, max_bootstrapped_demos=4, max_labeled_demos=16,
  max_rounds=1, teacher_settings=...)` — runs a teacher (defaults to the
  program itself) to generate full multi-stage demos, keeps those passing the
  metric. For ~10 examples.
- `BootstrapFewShotWithRandomSearch(metric, num_candidate_programs=16, ...)` —
  the above over multiple shuffles, pick best on valset. For 50+ examples.
  Alias `dspy.BootstrapRS`.
- `KNNFewShot(k, trainset, vectorizer=dspy.Embedder(...))` — retrieves nearest
  examples per input at inference time.

**Instruction optimization**

- `MIPROv2(metric, auto="light"|"medium"|"heavy", prompt_model=None,
  teacher_settings=None, num_threads=...)` — jointly optimizes instructions and
  demos via Bayesian search. The workhorse. 200+ examples recommended for long
  runs. Zero-shot mode: `compile(..., max_bootstrapped_demos=0,
  max_labeled_demos=0)`.
- `SIMBA(metric, bsize=32, max_steps=8, max_demos=4)` — introspects on
  high-variance minibatch failures and appends self-generated rules or demos.
  Strong on hard agent tasks.
- `COPRO(metric, breadth=10, depth=3)` — older coordinate-ascent instruction
  refinement.
- `GEPA` — reflective evolution; see below.

**Weight optimization**

- `BootstrapFinetune(metric=None, num_threads=...)` — distills a program into
  finetuned weights for a small LM. Requires `dspy.settings.experimental =
  True`. Pattern: build `student = program.deepcopy(); student.set_lm(small)`,
  `teacher = program.deepcopy(); teacher.set_lm(big)` (optionally
  MIPROv2-optimize the teacher first), then `optimizer.compile(student,
  teacher=teacher, trainset=...)`. Passing a metric filters bad trajectories
  out of the training data — a big quality boost. Local students use
  `LocalProvider` and must be `.launch()`ed (`.kill()` to free the GPU).
  Reference: AlfWorld 15% → 72%; a finetuned 1B beat its GPT-4o-mini teacher.

**Meta**

- `Ensemble(reduce_fn=dspy.majority, size=None)` — `compile([prog1, prog2,
  ...])`, e.g. top-5 candidates from a random-search run.
- `BetterTogether(metric, p=..., w=...)` — sequences prompt and weight
  optimizers (`strategy="p -> w -> p"` often beats either alone).
- RL optimizers (ArborGRPO) exist but are explicitly experimental, GPU-heavy,
  and per the docs "typically worse on cost/quality than MIPROv2 or SIMBA" —
  a last resort.

**Choosing**: ~10 examples → BootstrapFewShot; 50+ → BootstrapRS; want 0-shot
prompts → MIPROv2 instruction-only; 200+ examples and budget for 40+ trials →
MIPROv2; want feedback-driven instruction evolution or have rich textual
feedback signals → GEPA; need a small/cheap deployed model → BootstrapFinetune
from a strong teacher. Optimizers compose: run MIPROv2 twice, or MIPROv2 →
BootstrapFinetune, or extract candidates into an Ensemble. Expect to iterate —
also on the metric and program structure, not just the optimizer.

## GEPA

GEPA evolves instructions by *reflecting in natural language* on execution
traces, so it thrives on metrics that return textual feedback, and it needs a
strong reflection model.

```python
def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    correct = gold.answer == pred.answer
    fb = "Correct." if correct else (
        f"Wrong: expected {gold.answer}. Full solution: {gold.solution}")
    return dspy.Prediction(score=float(correct), feedback=fb)

optimizer = dspy.GEPA(
    metric=metric_with_feedback,
    auto="light",                      # exactly ONE of auto / max_full_evals / max_metric_calls
    reflection_lm=dspy.LM("openai/gpt-4.1", temperature=1.0, max_tokens=32000),
    num_threads=16, track_stats=True,
)
optimized = optimizer.compile(program, trainset=trainset, valset=valset)
print(optimized.predict.signature.instructions)   # inspect the evolved prompt
```

Hard requirements:

- The metric must accept **5 args** `(gold, pred, trace, pred_name,
  pred_trace)` — GEPA raises `TypeError` otherwise. Return a float or
  `dspy.Prediction(score=..., feedback=...)`; a bare float works but starves
  the reflection loop.
- Exactly one budget arg: `auto` ("light"/"medium"/"heavy"),
  `max_full_evals`, or `max_metric_calls`.
- `reflection_lm` is required and should be a strong model (temperature 1.0,
  large max_tokens); the task LM can stay small/cheap.
- Data split: standard ML convention (big trainset), unlike other prompt
  optimizers.

Feedback recipe: your metric usually already has the ingredients — expose
them. Name the gold label, enumerate what was correctly/incorrectly
included/missed, decompose aggregate scores per objective, include the
reference solution on failure. `pred_name`/`pred_trace` are set when GEPA asks
about a specific predictor; you can dispatch per-module feedback on
`pred_name` or return program-level feedback regardless.

`track_stats=True` exposes `optimized.detailed_results` (candidates, Pareto
scores, best outputs); `log_dir` enables checkpoint/resume. For inference-time
search: `valset=eval_batch, track_stats=True, track_best_outputs=True`.

## Tracking runs

```python
mlflow.set_experiment("dspy-opt")
mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)
```

Parent run = the whole optimization; child runs = candidate programs. Set
`log_traces_from_compile=False` on large datasets (memory).

## Saving optimized programs

```python
optimized.save("program.json")                 # state-only: readable, diffable, safe
fresh = MyProgram(); fresh.load("program.json")  # must recreate same architecture

optimized.save("./artifact/", save_program=True)  # whole program via cloudpickle
loaded = dspy.load("./artifact/")                 # no class needed; trusted files only
```

Use `.pkl` state (`allow_pickle=True` on load) only when state contains
non-JSON-serializable objects. Pin the DSPy version across save/load for
dspy<3; 3.x guarantees compatibility within a major version.
