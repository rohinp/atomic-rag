"""
LLMFaithfulnessScorer — measures whether the answer is grounded in the context.

Algorithm (matches Ragas faithfulness internals):
  Step 1 — claim extraction: ask the LLM to list every factual claim
            in the answer, one per line.
  Step 2 — claim verification: for each claim, ask whether it is
            supported by the context (YES/NO).
  Score   = supported_claims / total_claims

A claim is "supported" if it can be directly verified from the context text.
Claims that go beyond what the context says — even if plausible — are counted
as unsupported, driving the score down.

Reference:
  Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
  (2023). https://arxiv.org/abs/2309.15217
"""

from __future__ import annotations

import re
import time

from atomic_rag.models.base import ChatModelBase
from atomic_rag.schema import DataPacket, EvalScores, TraceEntry

from .base import PipelineEvalBase

_CLAIM_EXTRACTION_TEMPLATE = (
    "List every distinct factual claim made in the following answer. "
    "Write one claim per line. Do not include opinions or hedges — only "
    "concrete, verifiable statements.\n\n"
    "Answer: {answer}\n\n"
    "Claims (one per line):"
)

_VERIFICATION_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Claim: {claim}\n\n"
    "Is this claim directly supported by the context above? "
    "Answer with a single word: YES or NO."
)

_YES_RE = re.compile(r"\bYES\b", re.IGNORECASE)


def _parse_claims(raw: str) -> list[str]:
    """Split LLM output into individual claims, stripping bullets/numbers."""
    lines = []
    for line in raw.splitlines():
        # strip leading bullets, dashes, numbers
        clean = re.sub(r"^\s*[\-\*\d\.\)]+\s*", "", line).strip()
        if clean:
            lines.append(clean)
    return lines


def _is_supported(response: str) -> bool:
    return bool(_YES_RE.search(response.strip()))


class LLMFaithfulnessScorer(PipelineEvalBase):
    """
    Scores faithfulness via two-step LLM reasoning: claim extraction then
    per-claim verification against the context.

    Parameters
    ----------
    chat_model:
        Any ChatModelBase (OllamaChat, OpenAIChat, …).
    claim_template:
        Override the claim extraction prompt.  Must contain ``{answer}``.
    verify_template:
        Override the claim verification prompt.  Must contain ``{context}``
        and ``{claim}``.
    """

    def __init__(
        self,
        chat_model: ChatModelBase,
        claim_template: str | None = None,
        verify_template: str | None = None,
    ) -> None:
        self._chat = chat_model
        self._claim_tmpl = claim_template or _CLAIM_EXTRACTION_TEMPLATE
        self._verify_tmpl = verify_template or _VERIFICATION_TEMPLATE

    def score(self, packet: DataPacket) -> DataPacket:
        t0 = time.monotonic()

        answer = packet.answer.strip()
        context = packet.context.strip()

        # Guard: can't score without both answer and context
        if not answer or not context:
            faith_score = 0.0
            claims: list[str] = []
            supported = 0
        else:
            # Step 1: extract claims
            claim_prompt = self._claim_tmpl.format(answer=answer)
            raw_claims = self._chat.complete(claim_prompt)
            claims = _parse_claims(raw_claims)

            if not claims:
                faith_score = 0.0
                supported = 0
            else:
                # Step 2: verify each claim against the context
                supported = 0
                for claim in claims:
                    verify_prompt = self._verify_tmpl.format(
                        context=context, claim=claim
                    )
                    verdict = self._chat.complete(verify_prompt)
                    if _is_supported(verdict):
                        supported += 1

                faith_score = supported / len(claims)

        duration_ms = (time.monotonic() - t0) * 1000

        new_scores = packet.eval_scores.model_copy(
            update={"faithfulness": round(faith_score, 4)}
        )
        entry = TraceEntry(
            phase="evaluation",
            duration_ms=round(duration_ms, 2),
            details={
                "metric": "faithfulness",
                "claims_extracted": len(claims),
                "claims_supported": supported,
                "score": round(faith_score, 4),
            },
        )
        return packet.model_copy(update={"eval_scores": new_scores}).with_trace(entry)
