
import os
import json

from dotenv import load_dotenv
from openai import OpenAI


# ==================================================
# GROQ CLIENT SETUP
# ==================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. Check your .env file."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-20b"


# ==================================================
# HELPER: CALL MODEL AND PARSE JSON
# ==================================================

def call_model(prompt):

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("The model returned an empty response.")

    # Remove Markdown code fences if the model adds them.
    content = content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0].strip()

    return json.loads(content)


# ==================================================
# DAY 2: EXTRACT JOB REQUIREMENTS
# ==================================================

def extract_job_requirements(job_description):

    prompt = f"""
You are a professional recruiting requirements analyst.

Extract the job requirements from the job description.

Return one valid JSON object with these exact fields:

{{
    "job_title": "",
    "required_skills": [],
    "preferred_skills": [],
    "education_requirements": null,
    "preferred_education_requirements": null,
    "years_of_experience": null,
    "qualifications": [],
    "certifications": []
}}

Rules:
- Separate required skills from preferred skills.
- Do not invent requirements.
- Preserve the meaning of the job description.
- Use null when education or experience is not specified.
- Use empty lists when no items are found.
- Keep preferred education separate from required education.

JOB DESCRIPTION:
{job_description}

Return JSON only.
"""

    return call_model(prompt)


# ==================================================
# DAY 3: CLASSIFY AND WEIGHT REQUIREMENTS
# ==================================================

def classify_requirements(requirements):

    prompt = f"""
You are a recruiting requirements classification agent.

Classify every requirement into one of these categories:

must-have:
Essential for the job.
Weight = 5.

equivalent:
A genuinely comparable skill or experience.
Weight = 4.

nice-to-have:
Helpful but not essential.
Weight = 2.

Rules:
- Never invent requirements.
- Required skills are must-have unless a genuine equivalent
  is identified.
- Preferred skills are nice-to-have.
- Preferred education is nice-to-have.
- Required education is must-have.
- Required years of experience are must-have.
- Do not invent skill equivalents.
- PostgreSQL can be an equivalent to SQL experience
  because PostgreSQL uses SQL.

Return JSON in this structure:

{{
    "classified_requirements": [
        {{
            "requirement": "",
            "category": "must-have",
            "weight": 5,
            "equivalent_to": null,
            "reason": ""
        }}
    ]
}}

REQUIREMENTS:
{json.dumps(requirements, indent=2)}

Return JSON only.
"""

    result = call_model(prompt)

    return result.get("classified_requirements", [])


# ==================================================
# DAY 4: DETECT AMBIGUITIES
# ==================================================

def detect_ambiguities(requirements):

    prompt = f"""
You are a recruiting clarification agent.

Review these job requirements.

Identify important missing or unclear information
that could affect candidate screening.

Return JSON in this structure:

{{
    "needs_clarification": false,
    "ambiguities": [
        {{
            "issue": "",
            "priority": "medium",
            "question": ""
        }}
    ]
}}

Priority definitions:

high:
Missing information could significantly change screening.

medium:
Additional detail would improve screening accuracy.

low:
Optional information.

Rules:
- Do not ask about information already provided.
- Do not ask unnecessary questions.
- Focus on meaningful screening ambiguities.
- If no important ambiguity exists, return false
  and an empty ambiguities list.

REQUIREMENTS:
{json.dumps(requirements, indent=2)}

Return JSON only.
"""

    return call_model(prompt)


# ==================================================
# DAY 4: UPDATE REQUIREMENTS WITH RECRUITER ANSWERS
# ==================================================

def update_requirements_with_clarifications(
    requirements,
    recruiter_answers
):

    updated = requirements.copy()

    # Add recruiter-provided technical skills.
    technical_skills = recruiter_answers.get(
        "technical_skills", []
    )

    if isinstance(technical_skills, str):
        technical_skills = [technical_skills]

    for skill in technical_skills:
        if skill not in updated["required_skills"]:
            updated["required_skills"].append(skill)

    # Add industry experience.
    industry = recruiter_answers.get("industry_experience")

    if industry:
        industry_requirement = f"{industry} industry experience"

        if industry_requirement not in updated["required_skills"]:
            updated["required_skills"].append(
                industry_requirement
            )

    # Update years of experience.
    years = recruiter_answers.get("years_of_experience")

    if years is not None:
        updated["years_of_experience"] = years

    # Update education while preserving preferred status.
    education = recruiter_answers.get("education")

    if education:
        education_text = str(education).strip()

        if "preferred" in education_text.lower():
            updated["preferred_education_requirements"] = (
                education_text.replace(" preferred", "")
                .replace("Preferred ", "")
                .replace(" preferred", "")
                .strip()
            )
            updated["education_requirements"] = None

        else:
            updated["education_requirements"] = education_text
            updated["preferred_education_requirements"] = None

    # Update certifications.
    certifications = recruiter_answers.get("certifications")

    if certifications:
        if isinstance(certifications, str):
            if certifications.lower() in [
                "none",
                "none required",
                "not required"
            ]:
                updated["certifications"] = []
            else:
                updated["certifications"] = [certifications]

        elif isinstance(certifications, list):
            updated["certifications"] = certifications

    # Update preferred skills if explicitly provided.
    preferred_skills = recruiter_answers.get("preferred_skills")

    if preferred_skills:
        if isinstance(preferred_skills, str):
            preferred_skills = [preferred_skills]

        updated["preferred_skills"] = preferred_skills

    return updated


# ==================================================
# DAY 5: COMPLETE REQUIREMENTS AGENT WORKFLOW
# ==================================================

def run_requirements_agent(
    job_description,
    recruiter_answers=None
):

    print("\nSTEP 1: Extracting job requirements...")

    requirements = extract_job_requirements(
        job_description
    )

    print("\nSTEP 2: Detecting ambiguities...")

    ambiguities = detect_ambiguities(requirements)

    # If clarification is needed, pause for recruiter input.
    if ambiguities.get("needs_clarification") and not recruiter_answers:

        return {
            "status": "needs_clarification",
            "extracted_requirements": requirements,
            "clarification_questions": ambiguities,
            "classified_requirements": None,
            "final_requirements": None
        }

    # Apply recruiter answers.
    if recruiter_answers:

        print("\nSTEP 3: Updating requirements with recruiter answers...")

        requirements = update_requirements_with_clarifications(
            requirements,
            recruiter_answers
        )

        # Clear the previous questions after answers are submitted.
        ambiguities = {
            "needs_clarification": False,
            "ambiguities": []
        }

    print("\nSTEP 4: Classifying and weighting requirements...")

    classified = classify_requirements(requirements)

    print("\nSTEP 5: Preparing final requirements...")

    return {
        "status": "complete",
        "extracted_requirements": requirements,
        "clarification_questions": ambiguities,
        "classified_requirements": classified,
        "final_requirements": requirements
    }


# ==================================================
# TESTING
# ==================================================

if __name__ == "__main__":

    sample_job_description = """
    We are looking for a Business Analyst to support
    business operations.

    The candidate should have strong analytical skills
    and experience working with business processes.
    """

    recruiter_answers = {
        "technical_skills": [
            "SQL",
            "Python",
            "Power BI"
        ],
        "years_of_experience": 3,
        "industry_experience": "Financial services",
        "education": "Bachelor's degree preferred",
        "certifications": "None required"
    }

    result = run_requirements_agent(
        sample_job_description,
        recruiter_answers
    )

    print("\nFINAL RESULT:")
    print(json.dumps(result, indent=2))
