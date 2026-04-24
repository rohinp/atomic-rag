# Research References

Every technique in atomic-rag is grounded in published research. This page collects the full citations, organised by the pipeline phase they support, with links to the test cases that demonstrate each solution.

---

## Foundational Papers

### RAG — the paradigm

**Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."**
*NeurIPS 2020.* [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

The paper that named and formalised RAG: a generator conditioned on retrieved documents rather than solely on parametric knowledge. Introduces the distinction between parametric memory (model weights) and non-parametric memory (a retrieval index). atomic-rag implements this pattern across all five phases.

---

### The Seven Failure Points — the motivating paper for this library

**Barnett et al. (2024). "Seven Failure Points When Engineering a Retrieval Augmented Generation System."**
[arXiv:2401.05856](https://arxiv.org/abs/2401.05856)

Catalogues the seven most common ways a RAG pipeline fails in production. Each phase of atomic-rag addresses one or more of these:

| Failure Point | Phase that addresses it | Key test |
|---|---|---|
| FP1 — Messy source documents (garbage in) | Phase 1 — Ingestion | [`test_ingestion.py → TestMarkdownChunker::test_header_is_kept_with_its_content`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_ingestion.py) |
| FP2 — Chunks too small/large for the query | Phase 1 — Ingestion | [`test_ingestion.py → TestMarkdownChunker::test_oversized_section_split_by_paragraphs`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_ingestion.py) |
| FP3 — Query not matching chunk style | Phase 2 — Query Intelligence | [`test_query.py → TestHyDEExpander::test_populates_expanded_queries`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |
| FP4 — Correct documents not retrieved | Phase 3 — Retrieval | [`test_retrieval.py → TestHybridRetriever::test_bm25_only_and_vector_only_docs_both_surface_in_hybrid_results`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_retrieval.py) |
| FP5 — Correct answer not in extracted context | Phase 4 — Context Compression | [`test_context.py → TestSentenceCompressor::test_relevant_sentence_buried_in_middle_is_retained`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_context.py) |
| FP6 — Answer not formed from context | Phase 5 — C-RAG | [`test_agent.py → TestAgentRunner::test_uses_fallback_on_low_score`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_agent.py) |
| FP7 — Incomplete answer | Phase 5 — C-RAG | [`test_agent.py → TestAgentRunner::test_generator_not_called_on_fallback`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_agent.py) |

---

### RAG Survey

**Gao et al. (2023). "Retrieval-Augmented Generation for Large Language Models: A Survey."**
[arXiv:2312.10997](https://arxiv.org/abs/2312.10997)

Comprehensive survey of RAG variants: naive RAG → advanced RAG → modular RAG. Covers the full landscape of retrieval strategies, augmentation techniques, and generation patterns. Useful for understanding where each atomic-rag component sits in the broader taxonomy.

---

## Phase 2 — Query Intelligence

### HyDE

**Gao et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels."**
[arXiv:2212.10496](https://arxiv.org/abs/2212.10496)

Hypothetical Document Embeddings (HyDE): instead of embedding the raw query, ask an LLM to write a *hypothetical* answer document, then embed that. The hypothesis lives in the same semantic space as real answer documents, so the nearest neighbours in the index are more relevant than those of the bare question string.

Implemented in `HyDEExpander`. Key tests:

| What it tests | Test |
|---|---|
| Hypothetical doc replaces raw query embedding | [`test_query.py → TestHyDEExpander::test_populates_expanded_queries`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |
| Exactly one expanded query produced | [`test_query.py → TestHyDEExpander::test_expanded_queries_has_exactly_one_entry`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |
| Falls back to original query on LLM error | [`test_query.py → TestHyDEExpander::test_fallback_to_original_on_error`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |
| Trace records hypothetical document length | [`test_query.py → TestHyDEExpander::test_trace_records_hypothetical_length`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |

---

### Multi-Query Expansion

**Jagerman et al. (2023). "Query Expansion by Prompting Large Language Models."**
[arXiv:2305.03653](https://arxiv.org/abs/2305.03653)

Prompts an LLM to generate N alternative phrasings of the same question. Each phrasing is used as a separate vector query, and the results are fused. Different phrasings activate different parts of the index — a question about "gradient descent optimisation" may miss chunks that discuss "weight update rules" unless the alternative phrasing surfaces them.

Implemented in `MultiQueryExpander`. Key tests:

| What it tests | Test |
|---|---|
| N alternative queries generated | [`test_query.py → TestMultiQueryExpander::test_n_queries_limits_alternatives`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |
| Original query included by default | [`test_query.py → TestMultiQueryExpander::test_original_included_by_default`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |
| Multiple embed calls issued (one per query) | [`test_query.py → TestHybridRetrieverQueryIntegration::test_multiple_embed_calls_with_expanded_queries`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |
| Trace records requested vs generated count | [`test_query.py → TestMultiQueryExpander::test_trace_records_requested_and_generated`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_query.py) |

---

## Phase 3 — Retrieval

### BM25

**Robertson & Zaragoza (2009). "The Probabilistic Relevance Framework: BM25 and Beyond."**
*Foundations and Trends in Information Retrieval.* [Publisher link](https://www.nowpublishers.com/article/Details/INR-019)

BM25Okapi: the standard probabilistic keyword retrieval function. Scores documents by term frequency (TF) with diminishing returns, inverse document frequency (IDF) to downweight common terms, and document-length normalisation. Complements vector search by matching exact tokens — function names, acronyms, version numbers, error codes — that embeddings blur together.

Implemented in `BM25Retriever`. Key tests:

| What it tests | Test |
|---|---|
| CamelCase identifiers tokenised and matched | [`test_retrieval.py → TestBM25Retriever::test_camelcase_identifier_matches_query`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_retrieval.py) |
| Punctuation in query ignored | [`test_retrieval.py → TestBM25Retriever::test_punctuation_in_query_ignored`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_retrieval.py) |
| Zero-score documents filtered out | [`test_retrieval.py → TestBM25Retriever::test_search_filters_zero_score_docs`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_retrieval.py) |

---

### Reciprocal Rank Fusion

**Cormack, Clarke & Buettcher (2009). "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods."**
*SIGIR 2009.* [ACM DL](https://dl.acm.org/doi/10.1145/1571941.1572114)

RRF combines ranked lists from multiple retrieval sources without requiring score normalisation. Each document is scored as the sum of `1 / (k + rank)` across all lists — a document appearing at rank 1 in both vector and BM25 results scores much higher than one appearing at rank 1 in only one list. The constant `k=60` is the empirically validated default from the paper.

Implemented in `reciprocal_rank_fusion` (used internally by `HybridRetriever`). Key tests:

| What it tests | Test |
|---|---|
| Doc in both lists outscores doc in one list | [`test_retrieval.py → TestRRF::test_doc_in_two_lists_outscores_doc_in_one`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_retrieval.py) |
| BM25-only and vector-only docs both surface | [`test_retrieval.py → TestHybridRetriever::test_bm25_only_and_vector_only_docs_both_surface_in_hybrid_results`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_retrieval.py) |
| Deduplication across lists | [`test_retrieval.py → TestRRF::test_deduplication_across_lists`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_retrieval.py) |
| Higher k reduces score gap between ranks | [`test_retrieval.py → TestRRF::test_higher_k_reduces_score_gap`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_retrieval.py) |

---

### Cross-Encoder Reranking

**Reimers & Gurevych (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks."**
*EMNLP 2019.* [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)

Cross-encoders jointly encode the query and each candidate document (rather than encoding them separately like bi-encoders). This allows full attention across both texts, producing more accurate relevance scores — at the cost of O(n) inference calls. Used as a second-stage reranker over the top-k hybrid results.

Implemented in `CrossEncoderReranker`. Key tests:

| What it tests | Test |
|---|---|
| Scores sorted descending after reranking | [`test_retrieval.py → TestCrossEncoderReranker::test_rerank_sorts_descending`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_retrieval.py) |
| Reranker called with (query, doc) pairs | [`test_retrieval.py → TestCrossEncoderReranker::test_predict_called_with_query_doc_pairs`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_retrieval.py) |
| Reranker result reflected in trace | [`test_retrieval.py → TestHybridRetriever::test_trace_records_reranked_true`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_retrieval.py) |

---

## Phase 4 — Context Compression

### Lost in the Middle

**Liu et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts."**
[arXiv:2307.03172](https://arxiv.org/abs/2307.03172)

Demonstrates that LLM performance on multi-document QA degrades significantly when the relevant document is placed in the middle of a long context window rather than at the beginning or end. Models answer correctly ~75% of the time when the relevant document is first, dropping to ~45% when it is position 10 of 20. `SentenceCompressor` mitigates this by filtering out low-relevance sentences before passing context to the LLM.

Implemented in `SentenceCompressor`. Key tests:

| What it tests | Test |
|---|---|
| Relevant sentence buried in middle is retained | [`test_context.py → TestSentenceCompressor::test_relevant_sentence_buried_in_middle_is_retained`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_context.py) |
| Low-similarity sentences dropped | [`test_context.py → TestSentenceCompressor::test_low_similarity_sentence_dropped`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_context.py) |
| Minimum sentences always retained | [`test_context.py → TestSentenceCompressor::test_all_low_similarity_still_keeps_min_sentences`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_context.py) |
| Reduction percentage recorded in trace | [`test_context.py → TestSentenceCompressor::test_reduction_pct_reflects_dropped_sentences`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_context.py) |

---

## Phase 5 — Corrective RAG

### C-RAG

**Yan et al. (2024). "Corrective Retrieval Augmented Generation."**
[arXiv:2401.15884](https://arxiv.org/abs/2401.15884)

Inserts a quality-evaluation step between retrieval and generation. An evaluator scores how well the retrieved context answers the query (0.0–1.0). If the score is below a threshold, a fallback message is returned instead of generating — preventing confident hallucinations when the retriever fails. Outperforms standard RAG on 4 of 4 knowledge-intensive QA benchmarks.

Implemented in `AgentRunner` + `LLMEvaluator`. Key tests:

| What it tests | Test |
|---|---|
| High eval score triggers generation | [`test_agent.py → TestAgentRunner::test_populates_answer_on_high_score`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_agent.py) |
| Low eval score triggers fallback (no hallucination) | [`test_agent.py → TestAgentRunner::test_uses_fallback_on_low_score`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_agent.py) |
| Generator not called on fallback (no LLM cost) | [`test_agent.py → TestAgentRunner::test_generator_not_called_on_fallback`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_agent.py) |
| Exact threshold boundary generates | [`test_agent.py → TestAgentRunner::test_threshold_boundary_exact_match_generates`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_agent.py) |
| Empty context scores 0.0 without LLM call | [`test_agent.py → TestLLMEvaluator::test_empty_context_returns_zero_without_llm`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_agent.py) |

---

## Evaluation

### RAGAS

**Es et al. (2023). "RAGAS: Automated Evaluation of Retrieval Augmented Generation."**
[arXiv:2309.15217](https://arxiv.org/abs/2309.15217)

Framework for reference-free evaluation of RAG pipelines. Defines three core metrics: **faithfulness** (are claims in the answer grounded in the context?), **answer relevance** (does the answer address the question?), and **context precision** (are the retrieved chunks actually useful?). atomic-rag implements faithfulness and answer relevance natively, with optional Ragas library integration for all three.

Implemented in `LLMFaithfulnessScorer`, `EmbeddingAnswerRelevance`, `RagasEvaluator`. Key tests:

| What it tests | Test |
|---|---|
| Faithfulness: all claims supported → score 1.0 | [`test_evaluation.py → TestLLMFaithfulnessScorer::test_populates_faithfulness_score`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_evaluation.py) |
| Faithfulness: partial support → fractional score | [`test_evaluation.py → TestLLMFaithfulnessScorer::test_partial_support_gives_fractional_score`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_evaluation.py) |
| Answer relevance via cosine similarity | [`test_evaluation.py → TestEmbeddingAnswerRelevance::test_identical_vectors_give_score_1`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_evaluation.py) |
| Ragas metric names mapped to EvalScores fields | [`test_evaluation.py → TestRagasEvaluator::test_maps_faithfulness_to_eval_scores`](https://github.com/rohinp/atomic-rag/blob/main/tests/test_evaluation.py) |
