
from collections import defaultdict


# ==================================================
# CONFIGURATION
# ==================================================

MIN_CANDIDATES = 5
MAX_CANDIDATES = 20
MAX_SEARCH_QUERIES = 5
MAX_SEARCH_ITERATIONS = 5


# ==================================================
# REQUIREMENT FORMAT COMPATIBILITY
# ==================================================

def get_requirement_text(item):
    """Support both the original and shared requirement formats."""
    return item.get("requirement") or item.get("text")


def is_must_have(item):
    """Identify required requirements in either format."""
    return (
        item.get("category") in ["must-have", "equivalent"]
        or item.get("kind") == "must"
    )


def is_nice_to_have(item):
    """Identify preferred requirements in either format."""
    return (
        item.get("category") == "nice-to-have"
        or item.get("kind") == "nice"
    )


# ==================================================
# EXTRACT SEARCH TERMS
# ==================================================

def extract_search_terms(classified_requirements):

    search_terms = []

    for item in classified_requirements:

        requirement = get_requirement_text(item)

        if requirement:
            search_terms.append(requirement)

        equivalent = item.get("equivalent_to")

        if equivalent:
            search_terms.append(equivalent)

        equivalents = item.get("equivalents", [])

        search_terms.extend(equivalents)

    return list(dict.fromkeys(search_terms))


# ==================================================
# GENERATE SEARCH QUERIES
# ==================================================

def generate_search_queries(classified_requirements):

    must_have = []
    nice_to_have = []

    for item in classified_requirements:

        requirement = get_requirement_text(item)

        if not requirement:
            continue

        if is_must_have(item):
            must_have.append(requirement)

        elif is_nice_to_have(item):
            nice_to_have.append(requirement)

    queries = []

    if must_have:
        queries.append(" ".join(must_have[:3]))

        for skill in must_have:
            queries.append(skill)

    if nice_to_have:
        queries.append(" ".join(nice_to_have[:3]))

    return list(dict.fromkeys(queries))[:MAX_SEARCH_QUERIES]


# ==================================================
# COLLECT AND DEDUPLICATE CANDIDATES
# ==================================================

def collect_candidates(search_results):

    candidates = {}

    for result in search_results:

        candidate_id = result.get("candidate_id")

        if not candidate_id:
            continue

        if candidate_id not in candidates:

            candidates[candidate_id] = {
                "candidate_id": candidate_id,
                "resume_chunks": [],
                "matched_queries": [],
                "scores": []
            }

        # Support both the original text field
        # and Noah's planned snippet field.
        chunk = result.get("text") or result.get("snippet")

        if chunk and chunk not in candidates[candidate_id]["resume_chunks"]:
            candidates[candidate_id]["resume_chunks"].append(chunk)

        query = result.get("query")

        if query and query not in candidates[candidate_id]["matched_queries"]:
            candidates[candidate_id]["matched_queries"].append(query)

        score = result.get("score")

        if score is not None:
            candidates[candidate_id]["scores"].append(score)

    return list(candidates.values())


# ==================================================
# ADAPTIVE SEARCH DECISIONS
# ==================================================

def decide_search_strategy(
    candidate_count,
    min_candidates=MIN_CANDIDATES,
    max_candidates=MAX_CANDIDATES
):

    if candidate_count < min_candidates:
        return {
            "decision": "broaden",
            "reason": "Too few candidates were found."
        }

    elif candidate_count > max_candidates:
        return {
            "decision": "narrow",
            "reason": "Too many candidates were found."
        }

    return {
        "decision": "stop",
        "reason": "The candidate count is within the target range."
    }


# ==================================================
# BROADEN SEARCH
# ==================================================

def broaden_search_queries(classified_requirements):

    queries = []

    for item in classified_requirements:

        requirement = get_requirement_text(item)

        if requirement:
            queries.append(requirement)

        if is_nice_to_have(item):
            continue

        queries.extend(item.get("equivalents", []))

    return list(dict.fromkeys(queries))


# ==================================================
# NARROW SEARCH
# ==================================================

def narrow_search_queries(classified_requirements):

    must_have = []

    for item in classified_requirements:

        if is_must_have(item):

            requirement = get_requirement_text(item)

            if requirement:
                must_have.append(requirement)

    if len(must_have) >= 2:
        return [" ".join(must_have)]

    return must_have


# ==================================================
# SEARCH ITERATION LIMIT
# ==================================================

def should_continue_search(
    iteration,
    max_iterations=MAX_SEARCH_ITERATIONS
):

    return iteration < max_iterations


# ==================================================
# MAIN SEARCH AGENT WORKFLOW
# ==================================================

def run_search_agent(requirements, search_function=None):

    if requirements.get("status") != "complete":

        return {
            "status": "needs_clarification",
            "message": "Requirements must be completed before searching.",
            "candidates": []
        }

    classified_requirements = requirements.get(
        "classified_requirements", []
    )

    if not classified_requirements:

        return {
            "status": "error",
            "message": "No classified requirements were provided.",
            "candidates": []
        }

    queries = generate_search_queries(classified_requirements)

    if not queries:

        return {
            "status": "error",
            "message": "No search queries could be generated.",
            "candidates": []
        }

    if search_function is None:

        return {
            "status": "awaiting_retrieval",
            "message": "Resume retrieval has not been connected yet.",
            "search_queries": queries,
            "candidates": []
        }

    search_results = []

    for query in queries:

        results = search_function(query, k=MAX_CANDIDATES)

        for result in results:

            if isinstance(result, dict):

                result = result.copy()
                result["query"] = query

                search_results.append(result)

    candidates = collect_candidates(search_results)

    candidates = candidates[:MAX_CANDIDATES]

    return {
        "status": "complete",
        "search_queries": queries,
        "total_candidates_found": len(candidates),
        "minimum_candidate_target": MIN_CANDIDATES,
        "candidates": candidates
    }
