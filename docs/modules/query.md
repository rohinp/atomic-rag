# Query Module

**Location**: `atomic_rag/query/`

Implements Phase 2: expands the raw user query before retrieval to improve recall. Two complementary strategies: HyDE (hypothetical document generation) and Multi-Query Expansion (alternative phrasings).

## Components

| Class | Role |
|---|---|
| `QueryExpansionBase` | Abstract interface — subclass to add a new expansion strategy |
| `HyDEExpander` | Generates one hypothetical document to use as the retrieval query |
| `MultiQueryExpander` | Generates N alternative phrasings and fuses their results via RRF |

See [HyDE](../techniques/hyde.md) and [Multi-Query Expansion](../techniques/multi-query-expansion.md) for research rationale, alternatives, and effectiveness data.

## DataPacket Contract

**Input**: `DataPacket` with `query` set.

**Output**: New `DataPacket` with:
- `expanded_queries` — list of strings; HybridRetriever uses each for a separate vector search
- `trace` — one `TraceEntry` appended (`phase="query_expansion"`)

Input packet is never mutated.

## HybridRetriever Integration

When `packet.expanded_queries` is non-empty, `HybridRetriever` runs vector search once per expanded query and fuses all result lists with RRF. The original query is always used for BM25. No changes to the retriever API are required.

## Minimal Working Examples

**HyDE:**

```python
from atomic_rag.query import HyDEExpander
from atomic_rag.models.ollama import OllamaChat
from atomic_rag.schema import DataPacket

expander = HyDEExpander(chat_model=OllamaChat(model="llama3.2:3b"))

packet = DataPacket(query="how does auth work?")
packet = expander.expand(packet)

print(packet.expanded_queries)
# ["The authentication system validates tokens using JWT signatures..."]
print(packet.trace[-1].details)
# {'strategy': 'hyde', 'expanded_count': 1, 'hypothetical_length': 143}
```

**Multi-Query Expansion:**

```python
from atomic_rag.query import MultiQueryExpander
from atomic_rag.models.ollama import OllamaChat

expander = MultiQueryExpander(chat_model=OllamaChat(), n_queries=3)
packet = expander.expand(DataPacket(query="auth flow"))

print(packet.expanded_queries)
# ["auth flow", "token validation pipeline", "user login sequence", "API credential check"]
print(packet.trace[-1].details)
# {'strategy': 'multi_query', 'requested': 3, 'generated': 3, 'expanded_count': 4}
```

## Configuration

### HyDEExpander

| Parameter | Default | Effect |
|---|---|---|
| `chat_model` | required | Any `ChatModelBase` (OllamaChat, OpenAIChat, …) |
| `prompt_template` | built-in | Override with `{query}` placeholder |
| `fallback_to_original` | `True` | On LLM error, return original query rather than raising |

### MultiQueryExpander

| Parameter | Default | Effect |
|---|---|---|
| `chat_model` | required | Any `ChatModelBase` |
| `n_queries` | `3` | Number of alternative phrasings to generate |
| `prompt_template` | built-in | Override with `{query}` and `{n}` placeholders |
| `include_original` | `True` | Prepend the original query to `expanded_queries` |
| `fallback_to_original` | `True` | On LLM error, return original query rather than raising |

## Choosing a Strategy

| Strategy | Best for | Latency |
|---|---|---|
| HyDE | Short or vague queries with vocabulary gap | 1 LLM call |
| MultiQuery | Queries where same concept has many phrasings | 1 LLM call (N outputs) |
| Both (combined) | Maximum coverage — use for critical retrieval paths | 2 LLM calls |
| Neither | Short, terminology-matched corpora | 0 LLM calls |

## Running Tests

```bash
pytest tests/test_query.py -v
```
