# Requirements Matrix

Every explicit requirement from the assignment `README.md` (sections
"Required capabilities" 1–7 and "README requirements" 1–10), mapped to an
implementation component, its deterministic behavior where practical, its
test strategy, the visible evaluation case(s) that cover it, and the risk
of getting it wrong. Component names refer to modules defined in
`architecture.md`.

## 1. Retrieval-Augmented Generation

| Requirement | Component | Deterministic behavior | Test strategy | Covering eval case(s) | Risk if wrong |
|---|---|---|---|---|---|
| Split and index the supplied Markdown documents | `rag/ingest.py` | Chunking by H2 heading is a pure, deterministic function of file content | Unit test: fixed doc → fixed chunk list/boundaries | (infrastructure, not directly asserted) | Bad chunk boundaries silently drop or merge policy clauses (e.g. splitting the return-window number from its unit) |
| Preserve front-matter metadata | `rag/ingest.py` | Every chunk carries the full parsed front matter of its source doc | Unit test: parse all 14 docs, assert required keys present per doc | `standard-return-window` (`required_sources`), all `required_sources` cases | Losing `status`/`policy_authority` makes the authority layer blind, collapsing to raw similarity ranking |
| Retrieve only relevant passages, not the whole corpus | `rag/retrieval.py` | Top-K retrieval with a fixed, configured K; whole corpus never concatenated into the prompt | Unit test asserts prompt token/char budget; retrieval function returns ≤K chunks | All retrieval-category cases | Full-corpus stuffing defeats "retrieve only relevant passages" and makes conflict/authority logic harder to reason about |
| Prefer authoritative, active documents over superseded/non-policy ones | `rag/authority.py` | Hard filter/boost: `status=="active" and policy_authority=="official"` before any embedding-score comparison | Unit test with synthetic candidates crossing every status/authority combination | `standard-return-window` (`forbidden_sources_as_authority`), `retrieved-prompt-injection` | An LLM shown doc 02/14 alongside doc 01 without an authority signal may cite the wrong window (this is literally the customer's reported bug #1) |
| Include source references (filename + heading) in every policy/product answer | `llm/prompts.py`, `validation/response_validator.py` | Evidence bundle carries `(filename, heading)` per authoritative chunk used; validator rejects a "policy answer" response missing a `sources` field | Deterministic assertion: response envelope `sources[]` non-empty whenever `tool == not_called` and answer makes a policy claim | All `required_sources` cases | Uncited claims are unverifiable and violate the explicit rubric criterion "retrieval quality and document precedence" |
| Avoid unsupported claims (groundedness) | `llm/prompts.py` (system prompt), `validation/response_validator.py` | System prompt instructs answering only from evidence bundle; validator does light claim-vs-evidence keyword checks as a backstop | Deterministic `must_not_include` / `must_include_concepts` assertions | `unsupported-country`, `no-lifetime-warranty`, `insufficient-information` | Hallucinated policy details are the single highest-weighted failure mode (25% "reliability, groundedness, abstention") |
| Say when supplied information is insufficient | `llm/prompts.py`, `handoff.py` | If evidence bundle is empty/off-topic after retrieval, deterministic fallback response + `handoff=true` | Deterministic: assert `handoff==true` and no fabricated fact | `insufficient-information` | Silently guessing an answer with no source is a groundedness failure and privacy/trust risk |
| Surface genuine conflicts between current authoritative sources | `rag/conflict.py`, `validation/response_validator.py` | Curated conflict registry keyed by topic tag; if ≥2 authoritative chunks with different `document_id` and no supersession relationship share a topic tag, conflict object is attached to the evidence bundle deterministically | Deterministic: assert both `required_sources` present, `must_not_silently_choose_one==true`, `handoff==true` | `genuine-active-source-conflict` | Silently picking one side of the 11-vs-12 conflict directly reproduces the customer's reported bug #1 |

## 2. Order lookup as a tool/function

| Requirement | Component | Deterministic behavior | Test strategy | Covering eval case(s) | Risk if wrong |
|---|---|---|---|---|---|
| Full `orders.json` never sent to the model | `orders/repository.py`, `orders/tool.py` | Repository is only accessible via `lookup(order_id) -> ToolResult`; no code path serializes the raw dataset into an LLM message | Static/code-review check + integration test asserting prompt payload never contains `internal` or `customer.email` substrings | `valid-order-lookup`, `order-data-privacy` | Leaking the whole file exposes every customer's PII and internal notes at once |
| Ask for order ID when missing | `orchestration/agent.py` | If message requires order context and no `order_id` is resolvable from input or session state, tool is not called; agent asks for it | Deterministic: assert `tool=="not_called_without_id"` | `missing-order-id` | Guessing an ID or a status without one directly reproduces customer bug #2 (invented order info) |
| Handle unknown and malformed IDs safely | `orders/normalize.py`, `orders/tool.py` | Explicit 3-way classification: normalizable / not-plausibly-an-ID / plausible-but-absent (see `corpus-analysis.md` §4) | Deterministic unit tests per class + one visible case for the "absent" class | `unknown-order` | Conflating "malformed" with "unknown" produces confusing or unsafe responses (e.g. calling a tool with garbage input) |
| Normalize harmless input differences (case, whitespace) | `orders/normalize.py` | Pure function `normalize(raw: str) -> NormalizedId \| None`, regex-based, unit-testable in isolation | Deterministic unit tests: `" ord-1007 "`, `"ORD1007"`, `"Ord-1007"` all → `ORD-1007` | *(original case — not in visible set, see `evaluation-plan.md`)* | Over-eager "normalization" that guesses distant IDs violates the explicit "do not guess" instruction |
| Use `status` as authoritative | `orders/sanitize.py`, `llm/prompts.py` | Response-composition rule keyed purely on `status` enum, not on presence/absence of other fields | Deterministic unit tests, one per status value (8 total) | `cancelled-order-stale-eta`, `shipped-without-eta` | Trusting stale `carrier`/`estimated_delivery` over `status` reproduces the stale-ETA failure mode explicitly called out in the data dictionary |
| Never invent a delivery estimate when unavailable | `orders/sanitize.py`, `validation/response_validator.py` | If `estimated_delivery is None`, the sanitized projection carries no date field at all — nothing for the LLM to extrapolate from | Deterministic: assert no date-like token in response for `ORD-1011`/`ORD-1012` | `shipped-without-eta` | Inventing a plausible-sounding date is a groundedness and trust failure |
| Never report stale delivery fields for cancelled/returned orders | `orders/sanitize.py` | For `status in {cancelled, returned}`, sanitizer omits `carrier`/`tracking_number`/`estimated_delivery` from the projection entirely (relies on `customer_safe_message` instead) | Deterministic: assert old ETA string absent from response | `cancelled-order-stale-eta` | Direct reproduction of the assignment's named stale-ETA scenario |
| Never expose email/address/internal notes/risk scores | `orders/sanitize.py` | Allowlist projection (see `security-model.md`) — forbidden fields structurally cannot reach the LLM | Deterministic: regex/substring assertions for email pattern, exact risk-score value, note text | `order-data-privacy` | Direct PII/confidentiality breach; heavily weighted in rubric (15% "tool use, data handling, and privacy") |
| Never claim a lookup happened when it did not | `orchestration/agent.py`, observability trace | Tool-call flag in the response envelope is set only by an actual repository call; the LLM cannot set it | Deterministic: cross-check trace `tool_calls` against claimed lookup in text for a case where lookup is withheld | `missing-order-id` | Reproduces customer bug #2 exactly |

## 3. Multi-turn conversation

| Requirement | Component | Deterministic behavior | Test strategy | Covering eval case(s) | Risk if wrong |
|---|---|---|---|---|---|
| Maintain relevant session context ("What about Canada?", "When will it arrive?") | `orchestration/session.py` | Explicit session-state object with `last_topic_tags`, `last_order_id`, bounded turn history — updated deterministically after each turn, not inferred fresh each time | Deterministic: assert second-turn tool args / retrieval topic reuse prior-turn entities without re-asking | `canada-multiturn` | Reproduces customer bug #3 (lost context) exactly |
| Don't carry unrelated details indefinitely / don't mix sessions | `orchestration/session.py` | Session keyed by explicit `session_id`; bounded history window (turn-count or token-budget based) | Unit test: two distinct `session_id`s never share state; a stale entity beyond the window is not reused | *(original case, see `evaluation-plan.md`)* | Cross-session leakage is a privacy defect (one customer's order context bleeding into another's session) |

## 4. Prompting and agent behavior

| Requirement | Component | Deterministic behavior | Test strategy | Covering eval case(s) | Risk if wrong |
|---|---|---|---|---|---|
| Treat user messages, retrieved passages, tool results as untrusted data | `llm/prompts.py` (structural templating) | Untrusted content is wrapped in clearly delimited blocks in the prompt template, structurally separated from the system instructions — a template concern, not a model-compliance hope | Unit test on prompt-assembly function: system instructions and untrusted blocks are non-overlapping string regions | `retrieved-prompt-injection` | If untrusted text is concatenated indistinguishably with instructions, the model is more susceptible to injection |
| Follow application instructions over retrieved-document instructions | `llm/prompts.py`, `validation/response_validator.py` | System prompt explicitly states retrieved/tool text is data-only; validator scans final answer for injected directives having been obeyed (e.g. "approved", secrets) | Deterministic `must_not_follow` assertions | `retrieved-prompt-injection` | Direct reproduction of customer bug #4 |
| Refuse to reveal system prompt/hidden instructions/secrets | `llm/prompts.py`, `validation/response_validator.py` | Explicit refusal instruction + validator blocklist pattern for system-prompt leakage in the output | Deterministic: assert response does not contain system-prompt fragments | *(original case, see `evaluation-plan.md`)*; related: `retrieved-prompt-injection` | System-prompt leakage exposes internal design and enables further injection |
| Use company content, not general model knowledge, for company-specific questions | `llm/prompts.py` | System prompt restricts company-specific answers to evidence-bundle content only; empty-evidence fallback is abstention, not general knowledge | Deterministic: `insufficient-information` type checks | `insufficient-information`, `unsupported-country` | General-knowledge answers about a fictional company are ungrounded by construction |
| Ask a concise clarifying question when required info is missing | `orchestration/agent.py` | Deterministic trigger: order-dependent question + no resolvable order ID → clarifying-question path | Deterministic: assert question asked, no invention | `missing-order-id` | Guessing instead of asking reproduces customer bug #2 |
| Recommend human handoff on conflict / insufficient data / failed action | `handoff.py` (deterministic rules sourced from doc 13) | Rule table: conflict detected → handoff; evidence empty → handoff; `status==exception` → handoff; unsupported action requested → handoff | Deterministic unit tests, one per rule | `genuine-active-source-conflict`, `insufficient-information`, `unknown-order`, `order-data-privacy` | Under-triggering handoff leaves the customer stuck with a wrong/incomplete answer; over-triggering degrades UX — both are testable via the deterministic rule table |
| Never claim refund/cancellation/replacement/address-change was completed | `handoff.py`, `llm/prompts.py` | The mock system has no such action implemented anywhere in code — there is structurally nothing for the LLM to "confirm" | Deterministic: assert absence of completion-claim language for any action request | *(original case, see `evaluation-plan.md`)* | This is the class of failure most damaging to customer trust in a real deployment |

## 5. Evaluation suite

| Requirement | Component | Deterministic behavior | Test strategy | Covering eval case(s) | Risk if wrong |
|---|---|---|---|---|---|
| Covers every supplied visible case | `evaluation/runner.py` + `evaluation/visible-cases.json` (copied verbatim) | Runner loads the file unmodified and executes every case | CI runs the full suite on every push | all 15 visible cases | Missing coverage is directly and easily checked by reviewers re-running the same file |
| ≥5 original cases | `evaluation/custom-cases.json` | Same schema as visible cases, run by the same runner | CI includes custom file by default | See `evaluation-plan.md` §Custom cases | Thin custom coverage under-tests exactly the paraphrase/combination behavior reviewers will probe |
| One documented command | `README.md`, `Makefile`/`pyproject.toml` script | Single command, e.g. `python -m evaluation.runner` or `make eval` | Verified by following README from a clean clone | n/a | Undocumented or multi-step eval commands are a direct README-requirements failure |
| Individual + category-level reporting | `evaluation/runner.py` | Reporter emits per-case pass/fail and per-category rollups (retrieval, groundedness, tool-use, tool-reliability, privacy, conversation, prompt-security, abstention, source-conflict) | Golden-file test on reporter output format | n/a | A single aggregate score hides exactly the category weighting the rubric cares about |
| Deterministic assertions wherever practical | `evaluation/runner.py` (assertion library) | Substring/regex/exact-match/tool-arg/source-set assertions; LLM-graded checks used only for the minority of "concept" assertions, never exclusively | Meta-test: assert assertion functions are pure and side-effect-free | n/a | Non-deterministic eval scoring makes regressions invisible and violates the explicit "not exclusively LLM-graded" requirement |
| Bug diary (≥3 failures, ≥1 beyond visible-case wording) | `README.md` (authored during Phase 2, informed by real runs) | Each entry: repro, root cause, fix, regression test added to `custom-cases.json` | The regression test itself is the proof | n/a | A fabricated/thin bug diary is easy for a reviewer to spot as not backed by real regression tests |
| Baseline vs final results | `evaluation/results/baseline.json`, `.../final.json` | Baseline run captured before authority/sanitizer hardening; final run captured at completion | Both files checked in; README references the delta | n/a | Without a real baseline, "improvement" is an unverifiable claim |

## 6. Basic observability

| Requirement | Component | Deterministic behavior | Test strategy | Covering eval case(s) | Risk if wrong |
|---|---|---|---|---|---|
| Debug trace with message, history, retrieved passages+scores, tool calls+sanitized results, final response, handoff, errors | `observability/trace.py` | One structured JSON record per turn, fixed schema, written regardless of debug-flag state (flag only controls whether it's *printed*) | Schema unit test against a fixed set of required keys | n/a (cross-cutting; verified implicitly by every eval case's trace output) | Missing/inconsistent trace fields make failures unreproducible during grading |
| Never log secrets or forbidden fields | `observability/trace.py` (redaction filter) | Same allowlist used by `orders/sanitize.py` is reused for trace serialization — one source of truth, not a second hand-maintained blocklist | Deterministic: run the full `orders.json` through the trace serializer, assert no forbidden field/value appears in output | `order-data-privacy` (extended to trace output) | Logging is a common, easily-missed leakage vector even when the response itself is clean |

## 7. Minimal interface

| Requirement | Component | Deterministic behavior | Test strategy | Covering eval case(s) | Risk if wrong |
|---|---|---|---|---|---|
| CLI/web/API sufficient; shows answer, sources, handoff flag | `interface/cli.py` (+ optional `interface/api.py`) | Response envelope always carries `answer`, `sources[]`, `handoff: bool` — the interface only renders these, never invents formatting that implies more happened | Manual/GIF-recorded walkthrough per README requirement 10 | n/a | Rubric explicitly states visual polish is not scored — time is better spent elsewhere; but a garbled or missing sources/handoff display would undercut clarity for the required demo GIF |

## 8. README requirements (assignment §"README requirements", items 1–10)

| Requirement | Where satisfied | Notes |
|---|---|---|
| 1. Setup/run from clean clone | `README.md` (Phase 2) | Must be verified literally from a fresh `git clone` in CI or manually before submission |
| 2. Env vars + `.env.example` | `.env.example` (Phase 2) | No real credentials ever committed |
| 3. Model/embedding/framework/storage choices | `README.md`, mirrors `architecture.md` §Technology choices | |
| 4. Short architecture explanation | `README.md` (condensed from `architecture.md`) | |
| 5. Eval run command | `README.md` | Same command referenced in requirements-matrix §5 |
| 6. Baseline + final results by category | `README.md` (from `evaluation/results/*.json`) | |
| 7. Bug diary | `README.md` | ≥3 entries, ≥1 beyond visible-case wording |
| 8. Known limitations / production improvements | `README.md` | Should reference the open questions in `implementation-plan.md` |
| 9. AI coding tools used + one wrong/incomplete AI suggestion | `README.md` | Must be filled in honestly during/after Phase 2 build, not fabricated in Phase 1 |
| 10. 2–4 min GIF/video demo | `README.md` (embedded) | Must show: one KB question w/ citations, one order lookup, one multi-turn exchange, one refusal/handoff, the eval suite running |
