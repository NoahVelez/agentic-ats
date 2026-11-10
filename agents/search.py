# ==================================================
# SEARCH AGENT
# ==================================================

from collections import defaultdict


# ==================================================
# CONFIGURATION
# ==================================================

MIN_CANDIDATES = 5
MAX_CANDIDATES = 20
MAX_SEARCH_QUERIES = 5


# ==================================================
# HELPER: EXTRACT SEARCH TERMS
# ==================================================

def extract_search_terms(classified_requirements):
    """
    Extract useful search terms from the classified
    job requirements.
    """

    search_terms = []

    for item in classified_requirements:

        requirement = item.get("requirement")

        if requirement:
            search_terms.append(requirement)

        equivalent = item.get("equivalent_to")

        if equivalent:
            search_terms.append(equivalent)

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(search_terms))


# ==================================================
# HELPER: GENERATE SEARCH QUERIES
# ==================================================

def generate_search_queries(classified_requirements):
    """
    Generate targeted queries using job requirements.
    """

    must_have = []
    nice_to_have = []

    for item in classified_requirements:

        category = item.get("category")
        requirement = item.get("requirement")

        if not requirement:
            continue

        if category in ["must-have", "equivalent"]:
            must_have.append(requirement)

        elif category == "nice-to-have":
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
# HELPER: COLLECT CANDIDATES
# ==================================================

def collect_candidates(search_results):
    """
    Combine results from multiple searches.
    Avoid duplicate candidates.
    """

    candidates = {}

    for result in search_results:

        candidate_id = result.get("candidate_id")

        if not candidate_id:
            continue

        if candidate_id not in candidates:

            candidates[candidate_id] = {
                "candidate_id": candidate_id,
                "resume_chunks": [],
                "matched_queries": []
            }

        chunk = result.get("text")

        if chunk and chunk not in candidates[candidate_id]["resume_chunks"]:
            candidates[candidate_id]["resume_chunks"].append(chunk)

        query = result.get("query")

        if query and query not in candidates[candidate_id]["matched_queries"]:
            candidates[candidate_id]["matched_queries"].append(query)

    return list(candidates.values())


# ==================================================
# MAIN: RUN SEARCH AGENT
# ==================================================

def run_search_agent(requirements, search_function=None):
    """
    Search for resumes matching the job requirements.

    Args:
        requirements: Completed output from Requirements Agent.
        search_function: Noah's resume retrieval function.

    Returns:
        Search results and candidate shortlist.
    """

    # Step 1: Validate requirements.
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

    # Step 2: Generate search queries.
    queries = generate_search_queries(classified_requirements)

    if not queries:
        return {
            "status": "error",
            "message": "No search queries could be generated.",
            "candidates": []
        }

    # Step 3: Check whether retrieval is connected.
    if search_function is None:
        return {
            "status": "awaiting_retrieval",
            "message": "Resume retrieval has not been connected yet.",
            "search_queries": queries,
            "candidates": []
        }

    # Step 4: Search the resume database.
    search_results = []

    for query in queries:

        results = search_function(query, k=MAX_CANDIDATES)

        for result in results:

            if isinstance(result, dict):

                result = result.copy()
                result["query"] = query

                search_results.append(result)

    # Step 5: Combine duplicate candidates.
    candidates = collect_candidates(search_results)

    # Step 6: Limit the candidate shortlist.
    candidates = candidates[:MAX_CANDIDATES]

    # Step 7: Return results.
    return {
        "status": "complete",
        "search_queries": queries,
        "total_candidates_found": len(candidates),
        "minimum_candidate_target": MIN_CANDIDATES,
        "candidates": candidates
    }
    