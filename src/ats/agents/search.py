
"""Adaptive Search Agent using the project's MCP retrieval tools."""

from __future__ import annotations

from typing import Any

from fastmcp import Client

from ats.config import Settings, get_settings
from ats.models import (
    CandidatePool,
    PoolStats,
    QueryRound,
    RequirementSet,
    SearchHit,
)

MIN_CANDIDATES = 10
MAX_CANDIDATES = 20
MAX_SEARCH_QUERIES = 10


def generate_search_queries(requirements: RequirementSet) -> list[str]:
    """Generate the initial query from must-have and nice-to-have requirements."""
    must_have = [
        item.text for item in requirements.requirements if item.kind == "must"
    ]
    nice_to_have = [
        item.text for item in requirements.requirements if item.kind == "nice"
    ]

    terms = must_have[:3] + nice_to_have[:2]
    return [" ".join(terms)] if terms else []


def broaden_search_queries(requirements: RequirementSet) -> list[str]:
    """Broaden search using requirements and their equivalents."""
    queries = []

    for item in requirements.requirements:
        queries.append(item.text)
        queries.extend(item.equivalents)

    return list(dict.fromkeys(query for query in queries if query))


def narrow_search_queries(requirements: RequirementSet) -> list[str]:
    """Narrow search to must-have requirements."""
    must_have = [
        item.text for item in requirements.requirements if item.kind == "must"
    ]

    if len(must_have) >= 2:
        return [" ".join(must_have)]

    return must_have


def decide_search_strategy(
    candidate_count: int,
    min_candidates: int = MIN_CANDIDATES,
    max_candidates: int = MAX_CANDIDATES,
) -> str:
    """Choose whether to broaden, narrow, or stop searching."""
    if candidate_count < min_candidates:
        return "broaden"

    if candidate_count > max_candidates:
        return "narrow"

    return "stop"


def _extract_result(tool_result: Any) -> Any:
    """Extract data from a FastMCP tool result."""
    if getattr(tool_result, "is_error", False):
        raise RuntimeError("An MCP search tool returned an error.")

    data = getattr(tool_result, "data", None)

    if data is not None:
        return data

    structured = getattr(tool_result, "structured_content", None)

    if isinstance(structured, dict):
        return structured.get("result", structured)

    raise RuntimeError("The MCP tool returned no usable data.")


def _parse_search_hits(raw_hits: Any) -> list[SearchHit]:
    """Validate search results against the shared SearchHit model."""
    if not isinstance(raw_hits, list):
        raise ValueError("search_resumes did not return a list of hits.")

    return [SearchHit.model_validate(hit) for hit in raw_hits]


def _parse_pool_stats(raw_stats: Any) -> PoolStats:
    """Validate pool statistics against the shared PoolStats model."""
    return PoolStats.model_validate(raw_stats)


async def run_search_agent(
    requirements: RequirementSet,
    mcp_client: Client,
    llm: Any = None,
    trace: Any = None,
    *,
    settings: Settings | None = None,
) -> CandidatePool:
    """Run adaptive resume search through the MCP tools."""

    # Search uses MCP retrieval and does not require an LLM.
    del llm, trace

    settings = settings or get_settings()

    if not requirements.requirements:
        raise ValueError("No requirements were provided for searching.")

    if settings.strong_hit is None:
        raise ValueError("STRONG_HIT must be configured before searching.")

    current_queries = generate_search_queries(requirements)

    if not current_queries:
        raise ValueError("No search queries could be generated.")

    all_hits: dict[tuple[str, str], SearchHit] = {}
    search_history: list[QueryRound] = []
    candidate_ids: list[str] = []

    for round_number in range(1, settings.max_search_rounds + 1):
        round_candidate_ids: list[str] = []

        for query in current_queries[:MAX_SEARCH_QUERIES]:
            raw_result = await mcp_client.call_tool(
                "search_resumes",
                {
                    "query": query,
                    "k": settings.max_pool_size,
                },
            )

            hits = _parse_search_hits(_extract_result(raw_result))

            for hit in hits:
                key = (hit.candidate_id, hit.chunk_id)
                all_hits[key] = hit

                if hit.candidate_id not in round_candidate_ids:
                    round_candidate_ids.append(hit.candidate_id)

                if hit.candidate_id not in candidate_ids:
                    candidate_ids.append(hit.candidate_id)

            if len(candidate_ids) >= settings.max_pool_size:
                candidate_ids = candidate_ids[:settings.max_pool_size]
                break

        search_history.append(
            QueryRound(
                round=round_number,
                query=" | ".join(current_queries[:MAX_SEARCH_QUERIES]),
                candidate_ids=round_candidate_ids,
            )
        )

        if not all_hits:
            strong_candidate_count = 0
        else:
            stats_result = await mcp_client.call_tool(
                "get_pool_stats",
                {
                    "hits": [
                        hit.model_dump() for hit in all_hits.values()
                    ]
                },
            )

            stats = _parse_pool_stats(_extract_result(stats_result))
            strong_candidate_count = stats.candidates_with_strong_hit or 0

        strategy = decide_search_strategy(
            strong_candidate_count,
            max_candidates=settings.max_pool_size,
        )

        if strategy == "stop":
            break

        if round_number == settings.max_search_rounds:
            break

        if strategy == "broaden":
            next_queries = broaden_search_queries(requirements)
        else:
            next_queries = narrow_search_queries(requirements)

        current_queries = next_queries[:MAX_SEARCH_QUERIES]

        if not current_queries:
            break

    return CandidatePool(
        candidate_ids=candidate_ids[:settings.max_pool_size],
        queries=search_history,
    )
