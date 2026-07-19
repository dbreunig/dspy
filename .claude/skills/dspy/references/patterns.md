# DSPy task-type → pattern cookbook

Canonical patterns distilled from the official tutorials. Pick the row that
matches the task, then adapt the snippet. All snippets assume
`dspy.configure(lm=dspy.LM(...))` has run.

| Task type | Pattern |
|---|---|
| Simple QA / generation | `dspy.Predict("question -> response")`; upgrade to `dspy.ChainOfThought` for a quality win |
| Classification | `Literal[...]` or `str`-Enum output field + `dspy.ChainOfThought` |
| Entity / info extraction | Class signature with `list[str]` or `list[PydanticModel]` outputs + CoT |
| RAG | Retriever = any Python callable; custom Module: retrieve in `forward`, answer with CoT |
| Multi-hop research | Custom Module, query-gen + note-taking CoTs in a Python loop |
| Agent with tools | `dspy.ReAct(sig, tools=[plain_functions])` |
| Agent, custom control | Hand-rolled loop around a CoT that picks the next tool |
| Multi-turn chat | `history: dspy.History = dspy.InputField()`; you append turns |
| Multi-stage pipeline | One custom Module; one signature per stage; stage N outputs are stage N+1 inputs |
| Output constraints | `dspy.BestOfN` / `dspy.Refine` with a `reward_fn` |
| Math / computation | CoT first; `dspy.ProgramOfThought` when execution beats reasoning |
| Big data / DataFrames | `dspy.RLM` (experimental) |
| Images / audio | `dspy.Image` / `dspy.Audio` field types with a capable LM |

## Classification (typed output)

```python
from typing import Literal

class Emotion(dspy.Signature):
    """Classify the emotion expressed in the sentence."""
    sentence: str = dspy.InputField()
    sentiment: Literal["sadness", "joy", "love", "anger", "fear", "surprise"] = dspy.OutputField()

classify = dspy.ChainOfThought(Emotion)
classify(sentence="i started feeling a little vulnerable").sentiment  # -> "fear"
```

`str`-Enums work the same way and travel well across a pipeline:

```python
class EmailType(str, Enum):
    ORDER_CONFIRMATION = "order_confirmation"
    SUPPORT_REQUEST = "support_request"

class ClassifyEmail(dspy.Signature):
    """Classify the type and urgency of an email based on its content."""
    email_subject: str = dspy.InputField()
    email_body: str = dspy.InputField()
    email_type: EmailType = dspy.OutputField()
    urgency: Literal["low", "medium", "high"] = dspy.OutputField()
```

## Extraction (structured outputs)

```python
class PeopleExtraction(dspy.Signature):
    """Extract contiguous tokens referring to specific people, if any, from a list
    of string tokens. Output a list of tokens; do not combine multiple tokens into
    a single value."""
    tokens: list[str] = dspy.InputField(desc="tokenized text")
    extracted_people: list[str] = dspy.OutputField()

extractor = dspy.ChainOfThought(PeopleExtraction)
```

Pydantic models as output element types are fully supported:

```python
class ExtractedEntity(BaseModel):
    name: str
    entity_type: str

class ExtractEntities(dspy.Signature):
    """Extract key entities from the email."""
    email_body: str = dspy.InputField()
    key_entities: list[ExtractedEntity] = dspy.OutputField()
```

## RAG

The retriever is just Python — `dspy.retrievers.Embeddings`, BM25, ColBERTv2,
or your own search API all plug in the same way:

```python
embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5)

class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)
```

Evaluate RAG with `dspy.SemanticF1(decompositional=True)` when no exact answers
exist. Tutorial reference results: 42% (CoT alone) → 55% (add retrieval) → 61%
(MIPROv2).

## Multi-hop research

State accumulates in typed fields fed back around a Python loop; both
sub-modules get optimized jointly:

```python
class Hop(dspy.Module):
    def __init__(self, num_docs=10, num_hops=4):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought("claim, notes -> query")
        self.append_notes = dspy.ChainOfThought(
            "claim, notes, context -> new_notes: list[str], titles: list[str]")

    def forward(self, claim: str):
        notes, titles = [], []
        for _ in range(self.num_hops):
            query = self.generate_query(claim=claim, notes=notes).query
            context = search(query, k=self.num_docs)
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(prediction.new_notes)
            titles.extend(prediction.titles)
        return dspy.Prediction(notes=notes, titles=list(set(titles)))
```

## Agents: dspy.ReAct

Tools are plain functions — DSPy introspects the name, type hints, and
docstring, so all three must exist and be accurate:

```python
def search_wikipedia(query: str) -> list[str]:
    """Return the top-5 result snippets for the query."""
    ...

def lookup_wikipedia(title: str) -> str:
    """Return the text of the Wikipedia page, if it exists."""
    ...

sig = dspy.Signature("claim -> titles: list[str]",
                     "Find all Wikipedia titles relevant to verifying the claim.")
react = dspy.ReAct(sig, tools=[search_wikipedia, lookup_wikipedia], max_iters=20)
pred = react(claim="...")
pred.titles       # typed outputs, extracted from the trajectory by a final CoT
pred.trajectory   # dict: thought_0, tool_name_0, tool_args_0, observation_0, ...
```

Notes that generalize:

- A class signature's docstring is where the agent's role/policy lives ("You
  are an airline customer service agent... decide the right tool to use").
- Pydantic models are fine as tool argument types. Avoid `datetime` args (LMs
  specify them badly) — use a small custom model instead.
- Tools may `raise ValueError("no such itinerary")` — the message becomes an
  observation the agent can recover from. Give agents an escape-hatch tool
  (e.g. `file_ticket`) for requests they can't handle.
- Bound methods work as tools (`self.memory.store`), not just top-level functions.
- MCP: `dspy.Tool.from_mcp_tool(session, tool)` for each server tool, then
  `await react.acall(...)` (MCP tools are async). LangChain:
  `dspy.Tool.from_langchain(tool)`; add
  `dspy.configure(allow_tool_async_sync_conversion=True)` to call async tools
  synchronously.
- ReAct contains two optimizable predictors (the loop and the final extractor);
  optimizers improve both.

## Agent with custom control flow

When you need step caps, timeouts, or per-example toolsets, ReAct is just a
pattern you can re-implement: a CoT that picks the next function, looped
manually, with a synthetic `finish` tool:

```python
class Agent(dspy.Module):
    def __init__(self, max_steps=5):
        self.max_steps = max_steps
        instructions = "For the final step, call finish(answer) to return the answer."
        self.react = dspy.ChainOfThought(dspy.Signature(
            "question, trajectory, functions -> next_selected_fn, args: dict[str, Any]",
            instructions))

    def forward(self, question, functions):
        trajectory = []
        for _ in range(self.max_steps):
            pred = self.react(question=question, trajectory=trajectory, functions=list(functions))
            if pred.next_selected_fn == "finish":
                return dspy.Prediction(answer=pred.args.get("answer", ""), trajectory=trajectory)
            result = safe_call(functions[pred.next_selected_fn], pred.args)  # wrap errors into dicts
            trajectory.append({"fn": pred.next_selected_fn, "args": pred.args, "result": result})
        return dspy.Prediction(answer="", trajectory=trajectory)
```

Return errors as observation data (`{"return_value": ..., "errors": ...}`)
rather than raising, so the agent can react. `dspy.SIMBA` is a good optimizer
for hard agent tasks like this.

## Multi-stage pipelines

One signature per stage, each wrapped in CoT; typed outputs of one stage are
typed inputs of the next; merge everything into a single Prediction:

```python
class EmailProcessor(dspy.Module):
    def __init__(self):
        self.classifier = dspy.ChainOfThought(ClassifyEmail)
        self.extractor = dspy.ChainOfThought(ExtractEntities)
        self.summarizer = dspy.ChainOfThought(SummarizeEmail)

    def forward(self, email_subject, email_body):
        cls = self.classifier(email_subject=email_subject, email_body=email_body)
        ents = self.extractor(email_body=email_body, email_type=cls.email_type)
        summ = self.summarizer(email_body=email_body, email_type=cls.email_type)
        return dspy.Prediction(email_type=cls.email_type, urgency=cls.urgency,
                               entities=ents.key_entities, summary=summ.summary)
```

Modules with matching signatures are interchangeable — e.g. swap a pipeline's
final CoT answerer for `dspy.ProgramOfThought(sig)` when computation helps.

## Multi-turn conversation

`dspy.History` is an input type; DSPy expands it into real user/assistant
turns, but *you* own appending:

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()

predict = dspy.Predict(QA)
history = dspy.History(messages=[])
while True:
    question = input()
    outputs = predict(question=question, history=history)
    history.messages.append({"question": question, **outputs})
```

Message dicts are keyed by your signature's own field names.

## Output constraints: BestOfN and Refine

The replacement for the removed `dspy.Assert`/`dspy.Suggest`. Reward function
takes `(args, pred)` and returns a float; `threshold` short-circuits:

```python
def one_word_answer(args, pred):
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

best_of_3 = dspy.BestOfN(module=dspy.ChainOfThought("question -> answer"),
                         N=3, reward_fn=one_word_answer, threshold=1.0)
refine = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)
```

`Refine` additionally generates LM feedback between attempts and injects it as
a hint. The reward function can itself be an LM judge (a small CoT module).
Both run attempts at `temperature=1.0` with distinct `rollout_id`s, so they
genuinely sample differently despite caching. `fail_count` controls how many
attempt errors are tolerated before raising.

## Computation: ProgramOfThought and RLM

```python
pot = dspy.ProgramOfThought("question -> answer: float")   # writes & runs Python in a sandbox
```

Use PoT when execution beats reasoning (e.g. `12!/sum(primes<30)` — CoT gets it
wrong, PoT gets it right). Requires Deno (Pyodide sandbox); no filesystem or
network inside.

`dspy.RLM` (experimental) drives a Python REPL over huge inputs the model never
sees whole — inputs live as sandbox variables with previews; useful for
DataFrame analysis. Custom objects implement the `dspy.SandboxSerializable`
hooks to enter the sandbox.

## Multimodal

```python
class DogPicture(dspy.Signature):
    """Output the dog breed of the dog in the image."""
    image: dspy.Image = dspy.InputField()
    answer: str = dspy.OutputField()

dspy.Predict(DogPicture)(image=dspy.Image.from_url("https://.../dog.jpg"))
```

`dspy.Image` works in inline signatures too (`"prompt, current_image:
dspy.Image -> feedback: str, matches: bool"`). `dspy.Audio` similarly
(`from_array(array, sampling_rate)`, `from_file`, or `Audio(data=b64,
format="wav")`) — needs an audio-capable LM; keep few-shot demos to 0–2 when
optimizing (audio tokens are expensive) and pass `data_aware_proposer=False`
to MIPROv2 (its dataset summarizer can't read audio).

## Using a big model to teach a small one

Run a small task LM, but let a strong LM write prompts and demos:

```python
optimizer = dspy.MIPROv2(metric=metric, auto="medium", num_threads=16,
                         prompt_model=gpt4o, teacher_settings=dict(lm=gpt4o))
optimized_small = optimizer.compile(program_running_small_lm, trainset=trainset)
```

Tutorial reference results: ReAct agent on Llama-3.2-3B went 8% → 42% recall
this way. To go further, distill with `dspy.BootstrapFinetune` (see
`optimization.md`). Per-module LM assignment uses `module.set_lm(lm)`;
derive student from teacher with `teacher = program.deepcopy()`.
