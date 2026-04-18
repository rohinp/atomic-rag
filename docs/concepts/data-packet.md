# DataPacket — The Inter-Module Contract

## Problem Statement

In a multi-phase pipeline, modules need to pass data to each other. The naive approach is to let each module define its own input/output format — a function takes a string, returns a list of tuples, the next one takes that list, etc. This breaks down fast:

- You cannot inspect what happened mid-pipeline without adding print statements everywhere.
- Swapping one module forces you to change the next module's input parsing.
- Testing a later phase requires manually constructing the exact output of all earlier phases.
- Tracing a bug means reconstructing what each module saw, which is impossible after the fact.

The `DataPacket` solves this by defining a **single, versioned object** that every phase reads from and writes to. All state lives in one place.

## How It Works

A `DataPacket` is a [Pydantic](https://docs.pydantic.dev/) model with fields that correspond to the output of each phase:

```
query (str)                   ← you set this; no phase ever changes it
  ↓
expanded_queries (list[str])  ← Phase 2 populates this
  ↓
documents (list[Document])    ← Phase 3 populates this (retrieved + scored)
  ↓
context (str)                 ← Phase 4 populates this (compressed)
  ↓
answer (str)                  ← Phase 5 populates this
  ↓
eval_scores (EvalScores)      ← Eval layer populates this
```

`trace (list[TraceEntry])` runs in parallel — every phase appends one entry with its timing and diagnostics.

### The Immutability Rule

**Phases never mutate the packet they receive.** They always return a copy:

```python
# ✅ correct
def run(self, packet: DataPacket) -> DataPacket:
    result = do_work(packet.query)
    return packet.model_copy(update={"answer": result})

# ❌ wrong — mutates in place
def run(self, packet: DataPacket) -> DataPacket:
    packet.answer = do_work(packet.query)
    return packet
```

This makes it safe to fork a packet (e.g. run two retrieval strategies and compare), and makes tests trivial — the input fixture is never corrupted.

## Developer Benefits

- **Debuggability**: At any point you can `print(packet.model_dump_json(indent=2))` and see the full pipeline state. No reconstructing intermediate outputs.
- **Testability**: To test Phase 4 (context compression), just construct a `DataPacket` with `documents` pre-filled and call the module. You don't need Phase 3 to have run.
- **Swappability**: Replacing ChromaDB with Qdrant only requires changing Phase 3's internals — the `DataPacket` it returns is identical.
- **Observability**: `packet.trace` is a complete audit log. Feed it directly to Langfuse for full-trace monitoring.

## Alternatives

| Approach | Trade-offs |
|---|---|
| **Function-to-function passing** (tuples, dicts) | Simple for 2 phases, breaks at 5+. No observability. |
| **Shared mutable state / context object** | Convenient but causes order-of-write bugs and makes parallel execution unsafe. |
| **Message queue between phases** | Correct for distributed systems, massive overkill for an in-process library. |
| **LangChain's `RunnablePassthrough`** | Hides what's being passed; hard to inspect; couples you to LangChain's execution model. |

`DataPacket` is the right level of abstraction: structured enough to be inspectable, simple enough that you can construct one in three lines.

## Schema Reference

### `Document`

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | auto UUID | Unique chunk identifier |
| `content` | `str` | required | The text of this chunk |
| `source` | `str` | required | File path, URL, or logical name |
| `chunk_index` | `int \| None` | `None` | Position within source document |
| `metadata` | `dict` | `{}` | Arbitrary parser-supplied key/values (page number, section header, etc.) |
| `score` | `float` | `0.0` | Relevance score set by retriever/reranker; higher = more relevant |

### `TraceEntry`

| Field | Type | Default | Description |
|---|---|---|---|
| `phase` | `str` | required | Phase name (e.g. `"retrieval"`) |
| `timestamp` | `str` | auto ISO 8601 | When this phase ran |
| `duration_ms` | `float` | required | Wall time in milliseconds |
| `details` | `dict` | `{}` | Phase-specific diagnostics |

### `EvalScores`

| Field | Type | Default | Description |
|---|---|---|---|
| `faithfulness` | `float \| None` | `None` | [0,1] — is the answer grounded in context? |
| `context_precision` | `float \| None` | `None` | [0,1] — is the gold doc ranked highly? |

### `DataPacket`

| Field | Type | Default | Populated by |
|---|---|---|---|
| `session_id` | `str` | auto UUID | Caller |
| `created_at` | `str` | auto ISO 8601 | Caller |
| `query` | `str` | required | Caller — never overwritten |
| `expanded_queries` | `list[str]` | `[]` | Phase 2 |
| `documents` | `list[Document]` | `[]` | Phase 3 |
| `context` | `str` | `""` | Phase 4 |
| `answer` | `str` | `""` | Phase 5 |
| `eval_scores` | `EvalScores` | empty | Eval layer |
| `trace` | `list[TraceEntry]` | `[]` | Every phase |

## Minimal Example

```python
import time
from atomic_rag import DataPacket, Document, TraceEntry

# Start a pipeline
packet = DataPacket(query="What caused the 2008 financial crisis?")

# Simulate Phase 3 adding retrieved documents
t0 = time.monotonic()
docs = [
    Document(content="The crisis was triggered by...", source="report.pdf", score=0.91),
    Document(content="Subprime mortgage lending...", source="report.pdf", score=0.78),
]
packet = packet.model_copy(update={"documents": docs}).with_trace(
    TraceEntry(phase="retrieval", duration_ms=(time.monotonic() - t0) * 1000, details={"top_k": 5})
)

# Inspect
print(packet.top_documents(k=1)[0].content)
# → "The crisis was triggered by..."

# Serialise for Langfuse / logging
print(packet.model_dump_json(indent=2))
```
