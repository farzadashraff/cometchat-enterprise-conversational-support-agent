# Evaluation Plan

This document originally recorded the Phase 1 *design* for the
evaluation suite, written before any code existed. It now records the
Phase 7 *as-built* strategy — what was actually implemented, why it
differs from the Phase 1 proposal where it does, and its honest
limitations. See `docs/architecture.md` §20 for the full Phase 7
narrative (bug diary, baseline vs. final results, genuine risks).

## 1. Command

```bash
python -m aster_row_agent.cli eval [--out results.json] [--label baseline|final|...]
```

Added as a third subcommand alongside the existing `ingest`/`chat`
(`src/aster_row_agent/cli.py`), rather than a separate `evaluation.runner`
entry point — Phase 6 already established `cli.py` as the project's one,
documented application entry point, and Phase 7's own brief says to
prefer an existing entry point over inventing a new one. `--label` is
purely descriptive (stored in the JSON output); `--out` writes a
machine-readable JSON artifact in addition to the console report. Exit
code is non-zero if any case fails, so the command is CI-gateable.

No `ANTHROPIC_API_KEY` or network access is required — see §3.

## 2. Package layout

`src/aster_row_agent/evaluation/`:

- `harness.py` — runs one case's messages against a **real** `Agent`
  (real `EvidenceAssembler` with real embeddings over the real
  knowledge base, real `OrderLookupService` over real `data/orders.json`),
  with only the LLM replaced by a scripted `FakeLLMClient`. Two thin
  recording subclasses (`RecordingOrderLookupService`,
  `RecordingEvidenceAssembler`) observe what was called without changing
  behavior, giving the evaluator a structural "trace" for tool-call and
  evidence-bundle assertions.
- `scripts.py` — per-case-id scripted LLM answers, authored by hand from
  the real knowledge-base text (see §3 for why).
- `assertions.py` — one pure, deterministic checker function per `expect`
  field (see §4).
- `runner.py` — loads case files, runs each case, evaluates its `expect`
  block, returns a structured `EvalReport`.
- `reporter.py` — console table + JSON serialization, careful never to
  echo matched sensitive substrings into a report (§9).

`evaluation/visible-cases.json` (repo root) is the assignment's supplied
file, copied verbatim and never edited. `evaluation/custom-cases.json`
(repo root) holds the 12 original cases (§5).

## 3. Why the core suite is fully deterministic (no live LLM)

The Phase 7 brief requires the core suite to run without a real API key
or network access, for CI/local repeatability. Concretely, this means
every case runs against a `FakeLLMClient` scripted with a hand-authored,
plausible answer (see `scripts.py`) rather than a live model call.

**What this does and does not test.** The real, load-bearing weight of
this suite is on the parts of the pipeline that are deterministic
regardless of model output: retrieval and authority gating (real
embeddings, real corpus — `EvidenceBundle.customer_citable_sources` is
asserted directly, not parsed out of prose), conflict detection, order
lookup and sanitization, routing, the pre-LLM handoff rules, and response
validation (including deliberately re-running certain cases with a
scripted *bad* answer to prove the validator's backstop actually fires —
see the `genuine-active-source-conflict` one-sided-citation check
exercised in `tests/unit/test_agent_orchestration.py`). These are
LLM-independent, fully reproducible, and are what actually prevents the
four customer-reported failure modes.

What this suite does **not** test is whether a real Anthropic model would
*phrase* an answer the way the script does. A case passing means "given a
plausible, well-behaved model output, the rest of the pipeline handles it
correctly" — not "the real model will say this." This is an explicit,
documented limitation, not an oversight; per Phase 7 §23, an optional
live-LLM smoke test was considered but not built, to avoid adding
network dependency, cost, and flakiness to a suite that should stay
runnable with zero setup — a judgment call consistent with "do not
introduce unnecessary infrastructure."

One assertion type partially closes this gap without a live call:
`llm_prompt_must_include` inspects the actual `LLMRequest.user_content`
sent to the (fake) model, so a case can verify the *application* actually
surfaced a signal (e.g. "this order ID looks invalid") to the model,
independent of what the scripted answer says back — this is what makes
`malformed-id-combined-with-policy` a genuine regression test for BUG-002
rather than one that would pass regardless of the underlying fix (this
gap was itself caught while building the suite — see the bug diary).

## 4. Assertion types (deterministic-first)

Implemented in `assertions.py`, one function per `expect` field:

| `expect` field | Mechanism | Determinism |
|---|---|---|
| `must_include` / `must_not_include` | Case-insensitive substring match on the final `AgentResponse.answer` | Fully deterministic |
| `must_include_concepts` | A curated dict mapping each concept phrase (verbatim from the case files) to a hand-authored regex of keyword alternatives | Deterministic, but the most brittle category — see below |
| `must_not_invent` | A curated dict mapping forbidden categories ("arrival date", "carrier", "tracking number", ...) to a detection regex (dates, known carrier names, etc.) | Deterministic |
| `required_sources` | Exact subset check against `AgentResponse.citable_sources` | Fully deterministic |
| `forbidden_sources_as_authority` | Checked against the real `EvidenceBundle.customer_citable_sources` recorded by `RecordingEvidenceAssembler` — the structural authority decision, not what the scripted model happened to cite | Fully deterministic, LLM-independent |
| `tool` / `tool_arguments` | Checked against `RecordingOrderLookupService.calls` (raw inputs actually passed to `.lookup()`), normalized via the real `normalize_order_id` for comparison | Fully deterministic |
| `llm_prompt_must_include` | Substring check against the actual `LLMRequest.user_content` sent (see §3) | Fully deterministic |
| `must_ask_for` | Keyword presence check on the answer | Deterministic |
| `must_refuse_to_disclose` | Checks `AgentResponse.handoff is True` (refusal-framing); the corresponding literal PII values are covered by `must_not_include` | Deterministic |
| `handoff` | Exact boolean match against `AgentResponse.handoff` | Fully deterministic |
| `must_not_silently_choose_one` | `handoff is True` (both required sources are checked separately via `required_sources`) | Deterministic |
| `must_not_follow` | Keyword absence check on the answer | Deterministic |

No case's full answer is ever compared byte-for-byte, and no LLM is ever
used as a judge — matching the assignment's explicit requirement.

`must_include_concepts` is the one deliberately-flagged exception to
"fully mechanical": a concept like "Canada is supported" cannot be
substring-matched against its own English description, so each of the 24
distinct concept phrases used across the visible cases is mapped by hand
to a keyword/regex alternative a correct answer could reasonably use.
This is authored per-case and is acknowledged as the most brittle
assertion category — it is never the *only* signal a case relies on;
`required_sources`/`forbidden_sources_as_authority` (evidence-bundle-level,
LLM-independent) carry the real weight for retrieval/authority/conflict
categories.

## 5. Categories

The 10 categories named in the Phase 7 brief map directly onto the
`category` field already used by both case files: `retrieval`,
`groundedness`, `authority` (source-conflict/supersession split further
into `authority` and `source-conflict`), `tool-use`, `privacy`,
`prompt-security`, `multi-turn`/`conversation`, `source-conflict`,
`abstention`, `tool-reliability`. The reporter's category summary groups
by this field and prints a PASS/FAIL/TOTAL row per category plus an
OVERALL row (see §7's example output).

## 6. Visible-case coverage

All 15 cases in `evaluation/visible-cases.json` are covered, unmodified,
1:1. No visible case is dropped, altered, or hardcoded around.

## 7. Original cases (13)

`evaluation/custom-cases.json` — each targets a gap the visible set does
not exercise (paraphrase, combination, multi-turn contamination/isolation,
or a normalization edge case), not a rephrasing of a visible case:

1. `paraphrased-policy-question` — "What happens if I decide I don't want
   the item anymore?" (retrieval, paraphrase of the return-window question)
2. `order-plus-policy-combo` — order ID + refund-eligibility question in
   one message (tool-use + groundedness combination)
3. `ambiguous-order-followup-no-id` — "Where is my order?" → "What's the
   ETA?", neither turn ever supplies an ID (tool-use; must not hallucinate
   an order)
4. `system-prompt-extraction-paraphrase` — an injection paraphrase
   targeting internal escalation process disclosure (prompt-security)
5. `superseded-policy-paraphrase` — asks about the return window in
   wording that could retrieve both current and legacy policy (authority)
6. `breeze-conflict-paraphrase` — asks about the Breeze Tumbler conflict
   without naming the known conflict directly (source-conflict)
7. `unsupported-vegan-attribute-standalone` — the vegan question asked
   with no prior turn (groundedness/abstention)
8. `unsupported-action-cancel-now` — a direct cancellation command
   (abstention; must not claim completion)
9. `session-isolation-order-context` — order context established in one
   session must not leak into a second, independent session (multi-turn
   / privacy)
10. `malformed-id-combined-with-policy` — **BUG-002 regression**: a
    malformed order-ID-shaped candidate combined with a policy question
11. `unrelated-topic-then-vegan-question` — **BUG-003 regression**: an
    unrelated prior topic must not "rescue" a genuinely insufficient
    follow-up question into a false ANSWERABLE disposition
12. `order-id-normalization-lowercase-whitespace` — casing/whitespace
    normalization exercised through the full application, not just the
    unit-level normalizer
13. `short-unrelated-question-after-conflict-topic` — **BUG-005
    regression** (found during Phase 8 hardening): a *short* unrelated
    question ("Which products are vegan?", 4 words) following a Breeze
    Tumbler conflict topic must not be contaminated by it — the same
    failure class as case 11, but not caught by that fix's word-count-
    only threshold

Three of these (10, 11, and 13) are regression tests for genuine bugs
found *while building and hardening this suite* — see the bug diary in
`docs/architecture.md` §20.

## 8. Multi-turn strategy

Cases are either single-session (`messages: [...]`, replayed in one
session, matching `visible-cases.json`'s own convention) or
multi-session (`sessions: [[...], [...]]`, each an independent fresh
session) — the latter is used only for `session-isolation-order-context`,
where the assertion cares about state in the *second*, independent
session after an *unrelated* first session ran. Assertions are always
evaluated against the last turn of the last session.

## 9. Evaluator security (Phase 7 §25)

- `check_must_not_include`/`check_must_not_follow` deliberately report
  only a *count* of matched forbidden strings in failure details, never
  the matched string itself — several `must_not_include` values in the
  supplied `visible-cases.json` fixture are literal customer PII (an
  email address, a street address), and a failure detail must not become
  a second place that PII leaks into, even in a local console/JSON
  artifact.
- The evaluator never reads a raw `RawOrderRecord` to construct an
  expected answer — `RecordingOrderLookupService` only ever returns the
  same `CustomerSafeOrder` the real pipeline would.
- Nothing in `evaluation/results-*.json` includes API keys, full prompts,
  or full conversation transcripts — only case ids, categories,
  pass/fail, and short, redacted assertion details.

## 10. Baseline vs. final methodology

Since this repository has no prior commits, the "baseline" (pre-Phase-7)
state was captured by temporarily reverting the three Phase 7 code fixes
(BUG-001/002/003 — see the bug diary), running the suite, and recording
the result as `evaluation/results-baseline.json`; the fixes were then
restored (verified byte-identical via the full pytest suite passing
before and after) and the suite re-run as
`evaluation/results-final.json`. Both files are checked in and are never
edited by hand.

**Phase 8 note**: Phase 8's mandated end-to-end smoke test found one
further genuine bug (BUG-005 — see the bug diary), fixed with a 13th
original case (`short-unrelated-question-after-conflict-topic`). This
case did not exist when the baseline was captured, so
`results-baseline.json` still reports 27 total cases while
`results-final.json` (and the current suite) reports 28 — the baseline
is preserved as an honest historical snapshot of the Phase 6→7
transition rather than retroactively extended to match.

## 11. Known limitations

- `must_include_concepts` is a hand-authored keyword approximation, not
  semantic entailment (§4) — it can pass a technically-off answer that
  happens to contain the right keywords, or fail a correct answer phrased
  unusually. It is always paired with a structural, LLM-independent check
  (`required_sources`/`forbidden_sources_as_authority`/`handoff`) for any
  case where that distinction actually matters.
- The suite never exercises live model phrasing quality (§3) — this is
  deliberate, not an oversight, and the honest tradeoff is documented
  there.
- One visible case, `final-sale-damaged-exception`, originally failed in
  both the baseline and final runs. This was BUG-004 in the bug diary
  (`docs/architecture.md` §20): doc 04's "must not promise...before a
  human review is completed" sentence lives in a heading the retriever
  does not select for this exact query, so no signal available at the
  time (message keywords or retrieved evidence) reliably distinguished
  "reporting an actual damaged item" from "asking about the policy in
  the abstract." Root-caused after the original evaluation and
  subsequently fixed in commit `430d9fb`, with a dedicated regression
  test added — the evaluation suite now passes 28/28, and this is no
  longer a current limitation (see `docs/architecture.md` §22 for the
  fix).
