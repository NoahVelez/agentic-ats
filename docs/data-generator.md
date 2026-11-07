# Synthetic data generator

`scripts/generate_data.py` builds the synthetic dataset described in [SPEC §6.1](./SPEC.md#61-synthetic-data-generator-scriptsgenerate_datapy):
job descriptions, a pool of resumes for each job, and an **answer key** that says which resumes should be
shortlisted and which line proves each requirement.

> Synthetic data for a class project. Not for real hiring decisions.

## Contents

- [Quick start](#quick-start)
- [The `data/` folder](#the-data-folder)
- [How it works](#how-it-works)
- [Using a real LLM](#using-a-real-llm)
- [Command-line options](#command-line-options)
- [Output files](#output-files)
- [Writing a JD seed](#writing-a-jd-seed)
- [Validation and the BM25 report](#validation-and-the-bm25-report)
- [Reproducibility and the cache](#reproducibility-and-the-cache)
- [Troubleshooting](#troubleshooting)
- [Code map and tests](#code-map-and-tests)

## Quick start

```powershell
uv sync
uv run python scripts/generate_data.py --offline
```

`--offline` uses built-in templates instead of an LLM. It needs no API key, makes no network calls, and
finishes in about a second. It writes to `data/generated/` by default:

```
Wrote 1 job(s), 50 resumes, and the answer key to data\generated (generator: offline)

BM25 ranks for jd_analytics_mgr (pool 50, top-N = 16):
  hidden_gem:      cand_0017=19, cand_0043=20, ...
  keyword_stuffer: cand_0035=1, cand_0046=2, ...
```

To keep a dataset separate from the default one, write somewhere else:

```powershell
uv run python scripts/generate_data.py --offline --out $env:TEMP\ats_data
```

## The `data/` folder

```
data/
├── jd_seeds/                   # inputs: hand-written JD seeds
│   ├── analytics_manager.yaml  # the template; the only committed seed
│   └── <your_jd>.yaml          # local seeds (gitignored)
└── generated/                  # outputs: written by the script (gitignored)
    ├── jobs/  resumes/  answer_key.json  datagen_report.json
```

- **Generated data is never committed.** It's rebuilt from the seeds, so each teammate, CI, and the
  Docker build runs the generator themselves. Delete `data/generated/` at any time.
- **Only the template seed is committed.** Any other `*.yaml` you add to `data/jd_seeds/` stays local
  but is still picked up by default. To share a new seed, force-add it (`git add -f`) or add a `!`
  rule for it in `.gitignore`.
- Offline output and the answer key are the same on every machine for the same `--seed`. LLM-written
  text is not (see [Reproducibility](#reproducibility-and-the-cache)).

## How it works

The rule that matters most: **the answer key is decided in code, never by the LLM.** The LLM only writes
filler text around proof lines that code chose in advance, so a bad LLM response can make a resume
read poorly but cannot make the answer key wrong.

For each candidate the generator runs three steps:

1. **Facts (deterministic).** One `random.Random(seed)` picks the archetype, which requirements are
   met, which proof bullet to use for each one, years of experience, and job titles and dates. A seeded
   Faker creates the name, email, phone, address, and date of birth. These go **only** into the
   resume's `pii` field.
2. **Render.** The renderer (LLM or offline templates) writes the summary, experience entries,
   education, skills, and projects. It is told which JD keywords it must not use. Code then inserts
   the proof bullets into specific experience entries and records where each one landed.
3. **Validate (deterministic).** The resume is checked (see [Validation](#validation-and-the-bm25-report)).
   If a check fails, the candidate is rendered again with the reasons added to the prompt. After 3
   failed attempts the whole run stops with an error.

After the whole pool is built, the generator ranks it with BM25 and reports where the hidden gems and
keyword stuffers landed.

### Archetypes

Each JD gets these archetypes, in shuffled order, so a candidate ID reveals nothing about its type.

| Archetype | Default count | Answer-key verdicts | Proof bullets come from | Expected |
|---|---|---|---|---|
| `clear_fit` | 6 | All must-haves met, at least one nice-to-have met | `jd_phrasings` | shortlist |
| `hidden_gem` | 6 | All must-haves met | `paraphrases` only | shortlist |
| `keyword_stuffer` | 6 | Everything missing | none; every JD keyword is put in the skills list | reject |
| `near_miss` | 6 | Exactly one must-have missing, the rest met | mix of `jd_phrasings` and `paraphrases` | reject |
| `career_changer` | 4 | Each must-have `partial` or `met`, at least one `partial` | `transferable` (partial) or `paraphrases` (met) | shortlist |
| `filler` | 22 | Everything missing | none; unrelated career | reject |

The **planted shortlist size** (N) is `clear_fit + hidden_gem + career_changer`, which is 16 by default.

Each archetype is drawn from a career field: the JD's own field for `clear_fit`, `hidden_gem`, and
`near_miss`; nursing, teaching, military, or logistics for `career_changer`; marketing or office
administration for `keyword_stuffer`; and cooking, construction, design, or retail for `filler`.
Career changers get transferable bullets that fit their field when the seed has one (a nurse gets the
charge-nurse bullet, not the platoon one).

## Using a real LLM

Install a provider SDK and set three environment variables. These are **separate** from the app's
`LLM_*` variables on purpose: the data should be written by a different model than the one that
evaluates it.

| Variable | Value |
|---|---|
| `DATAGEN_LLM_PROVIDER` | `openai` or `anthropic` |
| `DATAGEN_LLM_MODEL` | Model name for that provider |
| `DATAGEN_LLM_API_KEY` | API key for that provider |

```powershell
uv sync --extra datagen-llm
$env:DATAGEN_LLM_PROVIDER = "openai"
$env:DATAGEN_LLM_MODEL    = "<model-name>"
$env:DATAGEN_LLM_API_KEY  = "<your-key>"
uv run python scripts/generate_data.py --seed 42
```

On macOS/Linux:

```bash
DATAGEN_LLM_PROVIDER=openai DATAGEN_LLM_MODEL=<model-name> DATAGEN_LLM_API_KEY=<your-key> \
  uv run python scripts/generate_data.py --seed 42
```

Never commit a key. `.env` is gitignored and `.env.example` lists the variable names without values.
Without `--offline`, the script exits with an error if any of the three variables is missing.

If the LLM returns something that doesn't parse into the expected JSON structure, the renderer asks
again up to 2 more times before counting that as a failed attempt.

## Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--seeds PATH [PATH ...]` | `data/jd_seeds` | Seed YAML files or directories (every `*.yaml` / `*.yml` in a directory is loaded) |
| `--out DIR` | `data/generated` | Output directory (gitignored) |
| `--seed N` | `42` | Random seed. Same seed gives the same facts and answer key |
| `--offline` | off | Use the template renderer; no LLM, no env vars needed |
| `--clear-fit N` | `6` | `clear_fit` resumes per JD |
| `--hidden-gem N` | `6` | `hidden_gem` resumes per JD |
| `--keyword-stuffer N` | `6` | `keyword_stuffer` resumes per JD |
| `--near-miss N` | `6` | `near_miss` resumes per JD |
| `--career-changer N` | `4` | `career_changer` resumes per JD |
| `--filler N` | `22` | `filler` resumes per JD |
| `--jaccard-threshold X` | `0.2` | Maximum word overlap allowed between a hidden-gem proof bullet and its requirement |
| `--max-attempts N` | `3` | Render attempts per candidate before the run fails |
| `--workers N` | `4` | Candidates rendered in parallel (does not change the output) |
| `--cache-dir DIR` | `.cache/datagen` | Where LLM responses are cached |
| `--no-cache` | off | Always call the LLM; don't read or write the cache |
| `-v`, `--verbose` | off | Log each rejected attempt and retry |

Examples:

```powershell
# A small pool for quick experiments
uv run python scripts/generate_data.py --offline --filler 5 --near-miss 3

# Only one seed file, different random seed
uv run python scripts/generate_data.py --offline --seeds data/jd_seeds/analytics_manager.yaml --seed 7
```

The SPEC minimum is at least 5 each of `clear_fit`, `hidden_gem`, `keyword_stuffer`, and `near_miss`,
and at least 3 `career_changer`. The defaults meet it.

## Output files

```
data/generated/
├── jobs/<job_id>.json        # JobDescription: job_id, title, text (JD markdown)
├── resumes/<candidate_id>.json
├── answer_key.json
└── datagen_report.json       # counts, regenerated candidates, BM25 ranks
```

Before writing, the script **deletes existing `*.json` files in `jobs/` and `resumes/`** of the output
directory, so a smaller rerun doesn't leave old candidates behind.

### Resume

```json
{
  "candidate_id": "cand_0001",
  "job_ids": ["jd_analytics_mgr"],
  "pii": { "name": "...", "email": "...", "phone": "...", "address": "...", "dob": "1993-10-22" },
  "summary": "Registered nurse with 6 years of experience, now moving into analytics.",
  "experience": [
    { "company": "...", "title": "Registered Nurse", "start": "2024-01", "end": "Present",
      "bullets": ["...", "Supervised 12 nurses across rotating shifts as charge nurse", "..."] }
  ],
  "education": [{ "institution": "...", "degree": "B.S.N.", "field": "Nursing", "year": 2019 }],
  "skills": ["ACLS", "patient care"],
  "projects": [{ "name": "...", "bullets": ["..."] }]
}
```

`pii` is the only place personal details appear. Ingest should drop it; no text field contains any of
it. Dates are `YYYY-MM`, the most recent role comes first, and it ends in `Present`.

### Answer key

```json
{
  "seed": 42,
  "generator": "offline",
  "jobs": [{
    "job_id": "jd_analytics_mgr",
    "requirements": [{ "id": "R1", "text": "People management experience", "kind": "must",
                       "weight": 3.0, "equivalents": ["people management", "..."] }],
    "shortlist_size": 16,
    "candidates": [{
      "candidate_id": "cand_0001",
      "archetype": "career_changer",
      "expected": "shortlist",
      "requirements": [{
        "requirement_id": "R1",
        "verdict": "partial",
        "proof": "Supervised 12 nurses across rotating shifts as charge nurse",
        "location": { "section": "experience", "entry_index": 1, "bullet_index": 4 }
      }]
    }]
  }]
}
```

- `verdict` is `met`, `partial`, or `missing`. `missing` has no `proof` or `location`.
- `proof` appears **verbatim** in the resume at
  `resume.experience[entry_index].bullets[bullet_index]`.
- `generator` is `offline` or `<provider>:<model>`, so you can tell which renderer built the data.
- The labelled requirements live only in the answer key. `jobs/*.json` holds the JD text, so agents
  have to work out the requirements themselves.

## Writing a JD seed

Seeds are hand-written YAML files in `data/jd_seeds/`. To add a JD, copy
[analytics_manager.yaml](../data/jd_seeds/analytics_manager.yaml) to a new file, give it a new
`job_id`, and edit it. New seed files are gitignored (see [The `data/` folder](#the-data-folder)).

```yaml
job_id: jd_analytics_mgr          # letters, digits, _ and - only; must be unique across seeds
title: Analytics Manager
field: analytics                  # career profile for fits/near-misses (see below)
text: |
  ...full JD markdown...
requirements:
  - id: R1
    text: People management experience
    kind: must                    # must | nice
    weight: 3
    jd_phrasings:                 # close to the JD's wording -> clear_fit, near_miss
      - "Managed a team of 6 data analysts, owning hiring, onboarding, and performance reviews"
    paraphrases:                  # same skill, different words -> hidden_gem, near_miss, career_changer (met)
      - "Led a 5-person analytics group, ran weekly 1:1s and quarterly reviews"
    transferable:                 # evidence from another field -> career_changer (partial)
      - "Supervised 12 nurses across rotating shifts as charge nurse"
    keywords: [people management, direct reports, managed a team, team lead, team leadership]
```

Aim for 6–8 requirements with at least 4 must-haves, and per requirement at least 3 `jd_phrasings`,
3 `paraphrases`, and 2 `transferable` bullets. More variety in each list means less repetition across
resumes.

### Seed rules (checked before generation)

The generator checks every seed first and stops with a list of problems if any rule is broken. These
rules keep the answer key honest:

| Rule | Why |
|---|---|
| Each `jd_phrasing` contains at least one of its own requirement's `keywords` | Otherwise a "JD-wording" fit isn't actually in JD wording |
| A `jd_phrasing` contains no keywords from **other** requirements | Otherwise one bullet would quietly prove two requirements |
| `paraphrases` and `transferable` bullets contain **no** JD keywords at all | Hidden gems and career changers must not be findable by keyword |
| Each paraphrase's word overlap with its requirement is below `--jaccard-threshold` | Keeps hidden gems hidden |
| No bullet looks like an email, phone number, or URL | No PII-like text in resumes |
| Requirement IDs are unique | |

Keyword matching is **case-insensitive and whole-phrase**: `SQL` matches "sql" and "T-SQL" but not
"PostgreSQL". `dashboard` does not match "dashboards", so list each form you care about (the example
seed lists both). Keywords may contain punctuation, like `A/B testing` or `C++`.

### Career fields

`field` must be a key in `FIELDS` in [templates.py](../src/ats/datagen/templates.py). Only `analytics`
exists for the JD side today. To add a JD in another field, add a `FieldProfile` for it (titles,
companies, generic bullets, skills, projects, degrees, summaries). The profile is used for job titles
and industry in the LLM prompt, and for all text in offline mode.

For career changers, write `transferable` bullets that mention the source field (for example
"nurses", "soldiers", "students", "warehouse") so they match the candidate's background. The
`markers` on each career-changer field profile decide which bullets fit which field.

## Validation and the BM25 report

Every resume must pass these checks. A failing resume is regenerated (up to `--max-attempts`), and if
it still fails, the run stops with an error naming the candidate and the problems.

| Check | Applies to |
|---|---|
| Every proof line appears verbatim in the resume, at its recorded location | all |
| No text field contains the candidate's name, email, phone, or street address | all |
| No text field matches an email, phone-number, or URL pattern | all |
| Keywords for requirements the candidate **doesn't** meet appear only in the skills list | `clear_fit`, `near_miss` |
| No JD keyword appears outside the skills list | `keyword_stuffer`, `filler`, `career_changer` |
| No JD keyword appears anywhere, including skills | `hidden_gem` |
| Every JD keyword is in the skills list | `keyword_stuffer` |
| Each proof bullet's word overlap with its requirement is below `--jaccard-threshold` | `hidden_gem` |

The overlap score is the Jaccard similarity between the bullet and the requirement's `text` plus
`keywords`, after lowercasing, removing common words, and trimming simple word endings.

### BM25 report

After the pool is built, the JD text is used as a BM25 query over all resumes in that JD's pool. The
console and `datagen_report.json` show the rank of every hidden gem and keyword stuffer.

- A hidden gem ranked inside the top N (the planted shortlist size) prints a **warning** but does not
  fail the run. It means that gem is easy to find by keyword, which weakens the test.
- Keyword stuffers should rank near the top. That's what makes them a useful trap for the keyword
  baseline.

## Reproducibility and the cache

- **Facts and the answer key** depend only on `--seed`, the seed files, and the archetype counts. They
  are the same in offline and LLM mode.
- **Offline output** is byte-identical for the same inputs, whatever `--workers` is set to.
- **LLM output** is cached in `.cache/datagen/` (gitignored), one file per response, keyed by
  `sha256(model + prompt)`. A rerun with the same seed, model, and prompts reads from the cache, makes
  no API calls, and produces byte-identical files.
- Changing the model, the seed, the counts, a seed file, or the prompt in
  [render.py](../src/ats/datagen/render.py) changes the prompts, so those candidates are rendered again.
- Use `--no-cache` to force fresh calls. Delete `.cache/datagen/` to clear the cache.

Neither the cache nor the generated data is in git, so two people who generate with an LLM will get
different resume text unless they share files. For a dataset everyone must agree on (for example the
final eval), generate it once and share `data/generated/` outside git, for example as a zipped
GitHub release asset, and record the seed, model, and commit with it.

## Troubleshooting

| Message | What to do |
|---|---|
| `missing env vars: DATAGEN_LLM_...` | Set the three variables, or add `--offline` |
| `provider 'openai' needs its SDK` | `uv sync --extra datagen-llm` |
| `JD seed problems: ...` | Fix the listed bullets. Usually a paraphrase uses a JD keyword, or a JD phrasing is missing its keyword |
| `cand_XXXX (...) failed validation after 3 attempts: ...` | Run with `-v` to see each attempt. If the LLM keeps using a forbidden word, try another model or raise `--max-attempts`. If offline mode fails, a template bullet or skill probably clashes with a new seed's keywords |
| `WARNING: ... hidden_gem cand_XXXX is BM25 rank ...` | Not fatal. Make that requirement's paraphrases share fewer words with the JD text |
| `duplicate job_id across seeds` | Give each seed file a unique `job_id` |

## Code map and tests

| File | Role |
|---|---|
| [scripts/generate_data.py](../scripts/generate_data.py) | CLI |
| [src/ats/datagen/facts.py](../src/ats/datagen/facts.py) | Fact sheets: archetypes, verdicts, proof choice, roles, PII |
| [src/ats/datagen/render.py](../src/ats/datagen/render.py) | LLM prompt, offline renderer, proof insertion |
| [src/ats/datagen/llm.py](../src/ats/datagen/llm.py) | `LLMClient` protocol, OpenAI/Anthropic clients, disk cache |
| [src/ats/datagen/validate.py](../src/ats/datagen/validate.py) | Seed rules, resume checks, BM25 report |
| [src/ats/datagen/pipeline.py](../src/ats/datagen/pipeline.py) | Runs facts → render → validate with retries |
| [src/ats/datagen/io.py](../src/ats/datagen/io.py) | Seed loading and output writing |
| [src/ats/datagen/templates.py](../src/ats/datagen/templates.py) | Career-field profiles and offline text |
| [src/ats/datagen/text.py](../src/ats/datagen/text.py) | Tokenizing, keyword matching, PII patterns |
| [src/ats/models.py](../src/ats/models.py) | `JobDescription`, `Requirement`, `Resume`, `AnswerKey` |

Tests use a fake LLM and never touch the network:

```powershell
uv run pytest -q tests/datagen
uv run ruff check .
uv run ruff format --check .
```
