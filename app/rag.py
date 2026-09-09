"""
Answer generation over retrieved context.

Two generators behind one interface:

  ClaudeGenerator     calls the Anthropic API with the retrieved chunks as context
                      and a system prompt that forbids answering beyond them.

  ExtractiveGenerator returns the strongest retrieved passages directly, with no
                      model call at all.

The extractive path is the default when no API key is present. It keeps the
service, its tests and CI runnable offline and free, and it means an API outage
degrades answer quality rather than taking the endpoint down.

Citations are attached by the service, not requested from the model, so they
cannot be hallucinated: every citation corresponds to a chunk that was actually
retrieved.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field

from .stores.base import Hit

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions using only the numbered context passages provided.

Rules:
- Use only what the passages state. Do not add outside knowledge.
- If the passages do not contain the answer, say so plainly. Do not guess.
- Cite the passages you used inline, like [1] or [2].
- Be concise and direct."""


@dataclass
class Citation:
    marker: int
    chunk_id: str
    doc_id: str
    source: str
    score: float
    excerpt: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    generator: str
    grounded: bool          # False when nothing was retrieved to ground on
    # Full text of every retrieved chunk. Used by the evaluation harness for
    # faithfulness scoring; deliberately not serialised into the API response,
    # where the truncated excerpts on each citation are what callers need.
    context_texts: list[str] = field(default_factory=list)


def _build_citations(hits: Sequence[Hit], excerpt_chars: int = 220) -> list[Citation]:
    return [
        Citation(
            marker=i + 1,
            chunk_id=h.chunk.id,
            doc_id=h.chunk.doc_id,
            source=h.chunk.source,
            score=round(h.score, 4),
            excerpt=h.chunk.text[:excerpt_chars]
            + ("…" if len(h.chunk.text) > excerpt_chars else ""),
            metadata=h.chunk.metadata,
        )
        for i, h in enumerate(hits)
    ]


def _format_context(hits: Sequence[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        title = h.chunk.metadata.get("title") or h.chunk.doc_id
        blocks.append(f"[{i}] (source: {h.chunk.source} · {title})\n{h.chunk.text}")
    return "\n\n".join(blocks)


class ExtractiveGenerator:
    """No model call. Returns the retrieved passages, ranked."""

    name = "extractive"

    def generate(self, query: str, hits: Sequence[Hit]) -> str:
        if not hits:
            return "No relevant passages were found for this question."
        lines = [
            "Answering from the retrieved passages "
            "(extractive mode, no language model configured):",
            "",
        ]
        for i, h in enumerate(hits, start=1):
            lines.append(f"[{i}] {h.chunk.text}")
            lines.append("")
        return "\n".join(lines).strip()


class ClaudeGenerator:
    """Anthropic-backed generation, constrained to the retrieved context."""

    name = "claude"

    def __init__(
        self, api_key: str, model: str = "claude-sonnet-4-6", max_tokens: int = 700
    ) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, query: str, hits: Sequence[Hit]) -> str:
        if not hits:
            return "No relevant passages were found for this question."

        prompt = (
            f"Context passages:\n\n{_format_context(hits)}\n\n"
            f"Question: {query}"
        )
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()


def get_generator(prefer: str = "auto"):
    """
    Resolve a generator. Falls back to extractive on a missing key or a failed
    client construction, rather than raising at startup.
    """
    if prefer in ("auto", "claude"):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            try:
                return ClaudeGenerator(api_key=key)
            except Exception as exc:
                logger.warning("claude generator unavailable (%s); using extractive", exc)
        elif prefer == "claude":
            logger.warning("claude requested but ANTHROPIC_API_KEY is unset; using extractive")
    return ExtractiveGenerator()


class RagPipeline:
    """Ties retrieval and generation together and attaches citations."""

    def __init__(self, agent, generator) -> None:
        self.agent = agent
        self.generator = generator

    def answer(
        self,
        query: str,
        k: int | None = None,
        sources: Sequence[str] | None = None,
    ) -> tuple[Answer, object]:
        result = self.agent.retrieve(query, k=k, sources=sources)

        try:
            text = self.generator.generate(query, result.hits)
            gen_name = self.generator.name
        except Exception as exc:
            # a provider failure must not fail the request
            logger.warning("generation failed (%s); falling back to extractive", exc)
            text = ExtractiveGenerator().generate(query, result.hits)
            gen_name = "extractive (fallback)"

        answer = Answer(
            text=text,
            citations=_build_citations(result.hits),
            generator=gen_name,
            grounded=bool(result.hits),
            context_texts=[h.chunk.text for h in result.hits],
        )
        return answer, result.plan
