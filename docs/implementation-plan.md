# Implementation Plan (Phase 2)

This is a proposed time-boxed sequence for the 6–8 hour build, aligned to
the rubric weights (reliability/groundedness/abstention 25%, retrieval/
precedence 20%, tool use/privacy 15%, evaluation/regression 20%,
multi-turn/observability 10%, code clarity 5%, README/demo 5%). Not
started yet — pending architecture sign-off (see §3 below).

## 1. Sequenced task breakdown

**Hour 0–0.5 — Scaffolding**
- Repo skeleton per `architecture.md` §14, `pyproject.toml`, `pytest`
  wired, `pydantic-settings` config + `.env.example`, copy
  `knowledge-base/` and `data/` verbatim from `reference/`.
- Domain models (`models.py`): `OrderRecord`, `CustomerSafeOrder`,
  `Chunk`, `EvidenceBundle`, `ConflictObject`, `ResponseEnvelope`,
  `Session`, `Turn`.

**Hour 0.5–1.5 — RAG ingestion + index**
- `rag/ingest.py`: parse front matter, chunk by H2, unit tests against
  all 14 real documents (assert exact expected chunk counts/headings).
- `rag/index.py`: BM25 index; add local embedding + cosine as time
  allows (fallback to BM25-only is acceptable, see §3).
- `rag/retrieval.py`: top-K search function, unit-tested with known
  queries against known expected top hits (e.g. "return window" → doc 01
  chunk ranks in top-K).

**Hour 1.5–2.5 — Authority, conflict, evidence**
- `rag/authority.py`: gate pipeline (supersession, authority, audience),
  one unit test per gate using synthetic + real chunks (must correctly
  exclude doc 02/14/13 from customer-citable results and include 01/09
  etc.).
- `rag/conflict.py`: curated topic-tag registry seeded with the one known
  pair (11 vs 12); unit test asserts the conflict is detected and that
  an unrelated pair of active docs (e.g. 05 vs 06) is not falsely flagged.
- `rag/evidence.py`: assemble bundle; unit test the "insufficient
  information" empty-bundle path.

**Hour 2.5–3.5 — Orders module**
- `orders/repository.py`: load `orders.json`, lookup by id.
- `orders/normalize.py`: 3-way classifier, unit tests for every scenario
  in `corpus-analysis.md` §4 (normalizable/malformed/unknown) plus all 12
  real order IDs round-tripping.
- `orders/sanitize.py`: allowlist projection + status-aware stale-field
  omission; unit test runs **every** order in `orders.json` through the
  sanitizer and asserts zero forbidden substrings (`internal`,
  `warehouse_note`, `risk_score`, `@example.test`, street-address tokens)
  appear anywhere in the serialized output — this single test is the
  strongest guardrail against a future regression.
- `orders/tool.py`: the callable tool definition wired to the above.

**Hour 3.5–4.5 — LLM client, prompting, orchestration**
- `llm/prompts.py`: system prompt (compiling doc 13's rules + the
  explicit behavioral constraints from docs 04/05/07/08/09/10), the
  delimited-block template.
- `llm/client.py`: provider wrapper, structured-output parsing.
- `orchestration/session.py`, `orchestration/agent.py`: turn decision
  logic, session state updates.
- First end-to-end manual smoke test (a few visible cases by hand).

**Hour 4.5–5.5 — Validation, handoff, observability**
- `validation/response_validator.py`: all checks from
  `architecture.md` §11.
- `handoff.py`: rule table, unit-tested per rule.
- `observability/trace.py`: schema + redaction (reusing sanitizer
  allowlist), unit test proving forbidden fields cannot appear in a
  trace even when fed a full raw order record.
- Interface (`interface/cli.py`): minimal REPL-style CLI showing answer/
  sources/handoff.

**Hour 5.5–7 — Evaluation suite + baseline/final runs**
- `evaluation/runner.py` + assertion library.
- Run against `visible-cases.json` unmodified; capture baseline
  (deliberately before authority/sanitizer hardening, per
  `evaluation-plan.md` §8, so this may mean running an early commit).
- Author the 8 proposed original cases in `evaluation/custom-cases.json`;
  iterate until the full suite (visible + custom) passes; capture final
  results.
- Fix real bugs found during this pass; write the bug diary entries as
  they occur (not retroactively invented).

**Hour 7–8 — README, demo, polish, CI**
- README: setup/run instructions verified from a literal clean clone,
  `.env.example`, architecture summary, eval command, baseline/final
  results table, bug diary, known limitations, AI-tool disclosure
  (including one wrong/incomplete AI suggestion encountered while
  building this), embedded demo GIF/video.
- `.github/workflows/ci.yml`: `pytest` + deterministic portion of eval
  suite on push.
- Final full-suite run to confirm the README's reported numbers match
  reality.

This is ~8 hours at full length; if time pressure hits, the first things
to cut are the optional FastAPI wrapper and the vector-embedding half of
retrieval (BM25-only remains a legitimate local index), never the
sanitizer, authority gating, or the evaluation suite.

## 2. Definition of done (per assignment + brief)

- Every visible case passes deterministically.
- ≥5 (delivered: 8 proposed) original cases pass.
- ≥3 real bug-diary entries with working regression tests.
- No forbidden order field ever appears in a prompt, response, or trace
  (proven by the whole-dataset sanitizer test in Hour 3.5–4.5).
- No filename-specific logic anywhere in `rag/authority.py` or
  `rag/conflict.py` — everything keyed on metadata fields.
- README runnable from a clean clone with no undocumented steps.

## 3. Ambiguities and risks requiring architecture review before Phase 2

1. **Conflict-detection generality.** The curated topic-tag/regex
   registry (`architecture.md` §5) only catches the one conflict present
   in this corpus (doc 11 vs 12) and any future pair someone
   remembers to add a tag/extraction rule for. A more general semantic-
   contradiction detector was considered and rejected as unreliable and
   too time-expensive for the timebox. **Needs sign-off**: is a curated,
   documented-as-narrow mechanism acceptable, or should time be spent on
   a more general (but weaker-guarantee) approach, e.g. asking the LLM
   itself to flag potential contradictions among the top-K authoritative
   chunks as a secondary, non-authoritative hint that a human reviews?
2. **Embedding/vector-index dependency risk.** Local embedding models
   (`fastembed`/`sentence-transformers`) add install weight and a small
   chance of environment friction inside the timebox. **Needs sign-off**:
   is BM25-only acceptable as the shipped default if the embedding
   dependency proves troublesome, with vector search noted as a
   documented limitation/future improvement, or is a real embedding
   component a hard requirement for the "RAG" label to be credible?
3. **LLM provider choice.** The assignment does not mandate a provider.
   This plan defaults to Anthropic Claude. **Needs confirmation** this is
   the intended provider (vs. OpenAI or another single provider) given
   API-key/billing availability for the actual build.
4. **Legacy-policy grandfathering.** Doc 01 states a 30-day window;
   doc 02 (superseded) stated 45 days pre-2026-04-01. Neither document
   explicitly says whether an order *placed* before 2026-04-01 but not
   yet past its return window today should still get 45 days. This
   plan's default: always answer with current policy (doc 01) unless the
   user explicitly frames the question as historical, and never silently
   apply doc 02's numbers to a live decision. **Needs confirmation** this
   is the intended behavior, since it's a genuine interpretive gap in the
   supplied corpus, not a documented rule.
5. **Doc 13 (internal escalation rules) retrieval status.** This plan
   compiles doc 13 into code rather than retrieving it live, but still
   indexes it (flagged non-citable) for observability. **Needs
   confirmation** this satisfies "split and index the supplied
   documents" as intended, versus an alternative where it's simply
   excluded from ingestion entirely.
6. **"Concept" assertion determinism.** `must_include_concepts` checks
   are implemented as authored keyword/regex sets per case rather than
   free-form semantic grading, per the explicit "not exclusively LLM-
   graded" requirement. This is inherently more brittle to paraphrase
   than true semantic grading. **Needs sign-off**: acceptable tradeoff,
   or should a secondary (non-authoritative) LLM-judge signal be added
   for these specific fields, reported separately and never overriding
   the deterministic result?
7. **Session/API scope.** No session-timeout or API surface is mandated
   by the assignment. Default: CLI-first with one session per process;
   FastAPI + explicit session TTL only if time remains. **Needs
   confirmation** this prioritization (reliability/eval work over
   interface breadth) matches expectations, consistent with the rubric's
   explicit "framework choice and quantity of code are not scoring
   criteria."

No implementation work has started; these seven points are the main
judgment calls this plan makes on the architect's behalf and are flagged
for review rather than silently decided.
