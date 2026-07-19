# DSPy in production: saving, caching, async, streaming, observability

## Saving and loading

**State-only** (recommended; JSON = readable, diffable, safe):

```python
program.save("program.json")                       # signature, demos, per-predict LM config
fresh = MyProgram()                                # must recreate the SAME architecture
fresh.load("program.json")
```

Use `.pkl` state only for non-JSON-serializable state (`dspy.Image`,
`datetime`): `fresh.load("program.pkl", allow_pickle=True)`. Loaded demos come
back as plain dicts, not `dspy.Example` — compare via `demo.toDict()`.

**Whole-program** (cloudpickle; directory path; no class reconstruction):

```python
program.save("./artifact/", save_program=True)
loaded = dspy.load("./artifact/")
# script-defined custom modules: save(..., modules_to_serialize=[my_module])
```

Pickle-based loading executes code — trusted files only. API keys are never
serialized; custom LM classes on load need `allow_unsafe_lm_state=True`.
Version rule: pin DSPy across save/load for <3.0; 3.x is compatible within a
major version. MLflow alternative: `mlflow.dspy.log_model(program, "model")` /
`mlflow.dspy.load_model(uri)`.

## Caching

In-memory LRU + on-disk cache are **on by default** (`~/.dspy_cache`, env
`DSPY_CACHEDIR`). Key = hash of the LiteLLM request minus credentials.

```python
dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)  # global off
lm = dspy.LM("openai/gpt-4o-mini", cache=False)                           # per-LM off
```

- Fresh-but-cached samples: pass a distinct `rollout_id` with `temperature >
  0` (`predict(..., config={"rollout_id": 5, "temperature": 1.0})`). With
  temperature 0 a rollout_id changes nothing.
- Cache hits report empty usage (`get_lm_usage()` → `{}`) — don't let that
  confuse cost accounting.
- Provider-side prompt caching (long static system prompts, ReAct):
  `dspy.LM("anthropic/...", cache_control_injection_points=[{"location":
  "message", "role": "system"}])`.
- Disable disk+memory cache in serverless (AWS Lambda) environments.

## Async

Built-in modules support `await module.acall(**kwargs)` natively. Custom
modules implement `aforward`:

```python
class MyProgram(dspy.Module):
    def __init__(self):
        self.fetch = dspy.ChainOfThought("question -> query")
        self.answer = dspy.ChainOfThought("question, context -> answer")

    async def aforward(self, question, **kwargs):
        query = (await self.fetch.acall(question=question)).query
        context = await fetch_context_async(query)
        return await self.answer.acall(question=question, context=context)

result = await MyProgram().acall(question="...")
```

For an existing sync program, `dspy.asyncify(program)` wraps it in a worker
thread pool (size `dspy.configure(async_max_workers=8)`) and propagates
`dspy.context` overrides. Async tools: `await tool.acall(...)`; from sync code
`with dspy.context(allow_tool_async_sync_conversion=True)`. `ReAct.acall()`
automatically awaits tools. Guidance: sync for prototyping, async for
high-throughput serving.

Thread rules: `dspy.configure` may only be re-called from the thread that first
called it; use `dspy.context(...)` anywhere else. `dspy.context` propagates
across `await` but not into bare threads — `dspy.Parallel` / `module.batch` /
`asyncify` snapshot it for you.

## Streaming

Wrap with `dspy.streamify`; listen per output field (field must be `str`):

```python
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
)

async for chunk in stream_predict(question="..."):
    if isinstance(chunk, dspy.streaming.StreamResponse):
        print(chunk.chunk, end="")          # token(s) for chunk.signature_field_name
    elif isinstance(chunk, dspy.Prediction):
        final = chunk                        # complete program output, always last
```

- One listener per field. Looped fields (ReAct's `next_thought`) need
  `allow_reuse=True`. Same field name in two predictors → disambiguate with
  `StreamListener(..., predict=program.predict1, predict_name="predict1")`.
- Cache hits emit no tokens, only the final Prediction — handle that path.
- Status updates ("calling tool..."): subclass
  `dspy.streaming.StatusMessageProvider` (hooks: `tool_start_status_message`,
  `module_end_status_message`, ...) and pass `status_message_provider=`;
  chunks arrive as `dspy.streaming.StatusMessage`.
- Sync consumption: `dspy.streamify(..., async_streaming=False)`.
- Program with `aforward`: pass `is_async_program=True`.
- FastAPI SSE: `from dspy.utils.streaming import streaming_response` →
  `StreamingResponse(streaming_response(stream), media_type="text/event-stream")`.

## Serving (FastAPI)

```python
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), async_max_workers=4)
program = dspy.asyncify(dspy.ChainOfThought("question -> answer"))

@app.post("/predict")
async def predict(q: Question):
    result = await program(question=q.text)
    return {"data": result.toDict()}
```

MLflow serving: `mlflow.dspy.log_model(program, "dspy_program",
task="llm/v1/chat")` then `mlflow models serve -m runs:/{run_id}/model`.
Caveat: MLflow needs positional args — wrap prebuilt modules in a custom
Module with `forward(self, messages)`.

## Observability & debugging

- `dspy.inspect_history(n=1)` — the exact prompts/responses of the last n LM
  calls. Limitations: LM calls only (no tools/retrievers), no latency metadata.
- MLflow tracing — full per-step traces including tools and retrievers, no
  account needed:

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("DSPy")
mlflow.dspy.autolog()
```

- Callbacks: subclass `dspy.utils.callback.BaseCallback` (`on_module_start/end`,
  `on_lm_start/end`, `on_tool_start/end`, `on_adapter_parse_start/end`, ...);
  register with `dspy.configure(callbacks=[...])`. Copy — don't mutate —
  inputs/outputs inside handlers.
- Usage/cost: `dspy.configure(track_usage=True)` → `pred.get_lm_usage()`;
  `sum(x["cost"] for x in lm.history if x["cost"])`.
- `dspy.configure(provide_traceback=True)` puts Python tracebacks in error
  logs; `logging.getLogger("dspy").setLevel(logging.DEBUG)` for verbosity.

Structured exceptions — all subclass `dspy.LMError` and carry `.model`,
`.provider`, `.status`, `.request_id`, `.retry_after`, `.code`:
`ContextWindowExceededError`, `LMRateLimitError`, `LMTimeoutError`,
`LMServerError`, `LMAuthError`, `LMInvalidRequestError`, `AdapterParseError`.
`dspy.is_retryable_lm_error(e)` classifies; `dspy.LM(num_retries=3)` already
retries with backoff via LiteLLM. On `ContextWindowExceededError` during
optimization: reduce `max_bootstrapped_demos`/`max_labeled_demos` or retrieved
passages.

## Parallel execution

```python
results = dspy.Parallel(num_threads=8)([(module, {"question": q}) for q in questions])
results = module.batch(examples, num_threads=8)     # examples need .with_inputs()
# with failures: module.batch(examples, return_failed_examples=True) -> (results, failed, excs)
```

Both snapshot `dspy.context` into workers. `dspy.Parallel` is a runner, not a
Module — it's invisible to optimizers and `save()`.

## Adapters

The adapter turns a signature + inputs into messages and parses the response.
Default is `ChatAdapter` (`[[ ## field ## ]]` markers — works on any model).
Configure globally or scoped:

```python
dspy.configure(adapter=dspy.JSONAdapter())          # native structured output; leaner
with dspy.context(adapter=dspy.XMLAdapter()): ...   # <field>value</field>
```

- `JSONAdapter(use_native_function_calling=True)` — best when the provider has
  a JSON/structured-output mode; avoid on small local models.
- `TwoStepAdapter(extraction_model=small_lm)` — reasoning models that format
  unreliably: main LM answers in prose, a cheap LM extracts fields.
- `BAMLAdapter` — try for deeply nested Pydantic outputs.
- ChatAdapter silently falls back to JSONAdapter on parse failure; in tests use
  `ChatAdapter(use_json_adapter_fallback=False)` to surface real errors.
- Debug what the LM actually sees: `adapter.format(signature, demos, inputs)`
  or just `dspy.inspect_history()`.
- Only ChatAdapter implements finetune-data export — use it with
  `BootstrapFinetune`.
