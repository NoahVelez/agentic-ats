"""Requirements Agent.

Retrieves a job description through MCP, then uses the shared LLM
client to produce structured hiring requirements or clarification questions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastmcp import Client
from pydantic import ValidationError

from ats.config import Settings, get_settings
from ats.llm import LLMClient, create_llm_client, extract_json_text
from ats.models import (
    ClarificationRequest,
    JobDescription,
    Question,
    RequirementSet,
)

PROMPT_PATH = Path(__file__).parent / "prompts" / "requirements_system.md"


def load_system_prompt() -> str:
    """Load the Requirements Agent system prompt."""
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Requirements system prompt not found: {PROMPT_PATH}")

    return PROMPT_PATH.read_text(encoding="utf-8")


def parse_model_response(response: str) -> dict[str, Any]:
    """Parse a JSON object returned by the LLM."""
    cleaned = response.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Claude may add a sentence before or after the JSON object.
        try:
            parsed = json.loads(extract_json_text(response))
        except json.JSONDecodeError:
            raise ValueError(f"Requirements Agent returned invalid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Requirements Agent response must be a JSON object.")

    return parsed


def build_user_prompt(
    job: JobDescription,
    max_questions: int,
) -> str:
    """Build the user prompt from the retrieved job description."""
    schema = {
        "job_id": job.job_id,
        "requirements": [
            {
                "id": "R1",
                "text": "A specific, testable requirement",
                "kind": "must",
                "weight": 3,
                "equivalents": [],
                "assumption": None,
            }
        ],
    }

    clarification_schema = {
        "questions": [
            {
                "id": "Q1",
                "text": "A concise question for the recruiter",
                "about_requirement": "R1",
            }
        ]
    }

    return f"""
Analyze this job description and extract structured hiring requirements.

Job ID: {job.job_id}
Job title: {job.title}

Job description:
{job.text}

Instructions:
- Extract specific, testable hiring requirements.
- Classify each requirement as "must" or "nice".
- Use weights from 1 to 3, where 3 is most important.
- Include reasonable equivalent skills or experience where appropriate.
- Record assumptions rather than inventing facts.
- Do not include personal characteristics or protected traits.
- If important information is too vague to evaluate fairly, ask the
  recruiter a clarification question.
- Ask no more than {max_questions} clarification questions.
- Return either a requirements object or a clarification object.
- Return valid JSON only, without explanatory text.

Requirements response format:
{json.dumps(schema, indent=2)}

Clarification response format:
{json.dumps(clarification_schema, indent=2)}

Maximum clarification questions: {max_questions}
""".strip()


def _get_job_from_mcp_result(result: Any) -> JobDescription:
    """Validate the job returned by the MCP get_job tool."""
    if getattr(result, "is_error", False):
        raise ValueError("The MCP get_job tool returned an error.")

    data = getattr(result, "data", None)

    if isinstance(data, list):
        data = data[0] if data else None

    if data is None:
        structured = getattr(result, "structured_content", None)

        if isinstance(structured, dict):
            data = structured.get("result", structured)

    if data is None:
        raise ValueError("The MCP get_job tool returned no job data.")

    # FastMCP can return a Pydantic model in result.data.
    # Convert wrapped model objects into plain data before validation.
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    elif hasattr(data, "root"):
        data = data.root
        if hasattr(data, "model_dump"):
            data = data.model_dump()

    try:
        return JobDescription.model_validate(data, from_attributes=True)
    except ValidationError as exc:
        raise ValueError("The MCP get_job tool returned invalid job data.") from exc


def _build_clarification_request(
    questions_data: list[Any],
    max_questions: int,
) -> ClarificationRequest:
    """Validate the model's questions and give them stable IDs Q1..Qn.

    IDs are always renumbered so they are unique and match the keys the
    recruiter answers with, whatever IDs the model chose.
    """
    questions = []

    for index, question_data in enumerate(
        questions_data[:max_questions],
        start=1,
    ):
        if not isinstance(question_data, dict):
            raise ValueError("Each clarification question must be an object.")

        try:
            question = Question.model_validate({**question_data, "id": f"Q{index}"})
        except ValidationError as exc:
            raise ValueError(f"Invalid clarification question: {exc}") from exc

        if not question.text.strip():
            raise ValueError("Clarification questions cannot be empty.")

        questions.append(question)

    return ClarificationRequest(questions=questions)


async def run_requirements_agent(
    job_id: str,
    mcp_client: Client,
    recruiter_answers: dict[str, Any] | None = None,
    *,
    llm_client: LLMClient | None = None,
    settings: Settings | None = None,
) -> RequirementSet | ClarificationRequest:
    """Run the Requirements Agent for a job.

    The job description is retrieved through MCP. The shared LLM client
    produces either a RequirementSet or a ClarificationRequest.
    """
    settings = settings or get_settings()
    llm_client = llm_client or create_llm_client(settings)

    # Retrieve the job through the MCP server.
    result = await mcp_client.call_tool(
        "get_job",
        {"job_id": job_id},
    )

    job = _get_job_from_mcp_result(result)

    # Build prompts.
    system_prompt = load_system_prompt()
    user_prompt = build_user_prompt(
        job,
        max_questions=settings.max_clarify_questions,
    )

    if recruiter_answers:
        user_prompt += (
            "\n\nRecruiter clarification answers:\n"
            + json.dumps(recruiter_answers, indent=2)
            + "\n\nUse these answers when finalizing the requirements."
            + " The recruiter has already answered; do not ask further"
            " questions. Return the requirements object, recording any"
            " remaining uncertainty in `assumption`."
        )

    # Call the shared asynchronous LLM client.
    response = await llm_client.complete(
        system_prompt,
        user_prompt,
    )

    parsed = parse_model_response(response)

    questions_data = parsed.get("questions")

    if questions_data is not None and not isinstance(questions_data, list):
        raise ValueError("Clarification questions must be provided as a list.")

    # An empty list (e.g. {"requirements": [...], "questions": []}) is not a
    # clarification request; it would leave the run waiting on nothing.
    if questions_data:
        if recruiter_answers:
            if "requirements" not in parsed:
                raise ValueError(
                    "Requirements Agent asked for clarification again "
                    "after the recruiter had answered."
                )
        else:
            return _build_clarification_request(
                questions_data,
                settings.max_clarify_questions,
            )

    # Validate and return the final requirements.
    parsed["job_id"] = job.job_id

    try:
        requirement_set = RequirementSet.model_validate(parsed)
    except ValidationError as exc:
        raise ValueError(f"Invalid Requirements Agent output: {exc}") from exc

    for index, requirement in enumerate(
        requirement_set.requirements,
        start=1,
    ):
        expected_id = f"R{index}"

        if requirement.id != expected_id:
            raise ValueError(
                "Requirement IDs must be sequential "
                f"(expected {expected_id}, got {requirement.id})."
            )

        if not 1 <= requirement.weight <= 3:
            raise ValueError(
                f"Requirement {requirement.id} has invalid weight "
                f"{requirement.weight}. Weight must be between 1 and 3."
            )

    return requirement_set
