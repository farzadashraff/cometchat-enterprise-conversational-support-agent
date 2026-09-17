# Architecture

## 1. Design principles

1. **Deterministic code owns business-critical rules.** Authority
   ranking, conflict detection, order-field sanitization, status-based
   response rules, the cancellation time-window check, and handoff
   triggers are all plain Python functions with unit tests — not
   LLM judgment calls. The LLM is used for language understanding
   (intent, entity extraction, follow-up resolution, phrasing the final
   answer from a pre-assembled evidence bundle), not for enforcing
   policy or security rules.
2. **Untrusted content structurally cannot become instructions.** See
   `security-model.md` for the full trust-boundary design; the short
   version is that the prompt template puts a hard structural wall
   between system instructions and anything sourced from documents,
   tool results, or the user, and the sanitizer makes forbidden order
   fields physically absent from anything the LLM receives.
3. **Smallest stack that is still "real."** A local lexical+vector
   hybrid index instead of a hosted vector DB; one LLM provider; no
   agent framework (LangChain/LlamaIndex/etc.) — the orchestration loop
   here is simple enough (one retrieval step + at most one tool call
   per turn) that a framework would add indirection without solving a
   real problem for this assignment's scope.
4. **Metadata-driven, not filename-driven.** No rule anywhere refers to
   a specific filename or document_id; everything is expressed over
   `status`, `policy_authority`, `audience`, `supersedes`/
   `superseded_by`, and a small curated `topic_tags`/conflict registry
   (documented as the one deliberately non-generalizing piece — see
   §7).

## 2. Module map

```
interface        → CLI (primary) [+ optional thin FastAPI]
orchestration     → turn loop: intent → retrieval/tool decision → generation → validation
  ├─ agent         (single-turn decision logic)
  └─ session       (per-session conversation + entity state)
rag               → everything document-related
  ├─ ingest        (load + chunk + attach metadata)
  ├─ index         (local hybrid BM25 + vector index)
  ├─ retrieval     (top-K candidate retrieval)
  ├─ authority     (status/policy_authority/audience gating + supersession)
  ├─ conflict      (curated topic-tag conflict registry)
  └─ evidence      (assemble the evidence bundle passed to the LLM)
orders            → everything order-related
  ├─ repository    (load orders.json once, in-memory lookup by id)
  ├─ normalize     (order-ID normalization / classification)
  ├─ sanitize      (allowlist projection to CustomerSafeOrder)
  └─ tool          (the order_lookup tool the LLM can call)
llm               → provider access
  ├─ client        (thin wrapper over one provider's SDK)
  └─ prompts       (system prompt + structural templating of untrusted blocks)
validation        → response_validator (post-generation deterministic checks)
handoff           → deterministic handoff rule table (sourced from doc 13)
observability     → trace (structured per-turn JSON record + redaction)
evaluation        → runner + assertion library + case files + results
config            → pydantic-settings, .env-driven
models            → shared Pydantic domain models used across all modules
```

Dependency direction is one-way: `interface → orchestration → {rag,
orders, llm, handoff} → {validation, observability} → models/config`.
Nothing in `rag` or `orders` imports from `llm`; the LLM never sees raw
domain objects, only bundles/projections built for it.

## 3. Turn-level data flow

```
user message
    │
    ▼
session.update_with_user_message()      (append to bounded history)
    │
    ▼
agent.decide_turn()
    ├─ resolve references using session state (e.g. "it" → last_order_id,
    │  "what about Canada?" → last_topic_tags)
    ├─ classify turn: order-lookup-needed? policy/product question?
    │  off-topic/insufficient? unsupported-action request?
    │
    ├── if order-lookup-needed:
    │     order_id = normalize.extract_and_normalize(message, session)
    │     if order_id is None:  → ask clarifying question, tool NOT called, done
    │     else: orders.tool.lookup(order_id) → sanitize.project() → ToolResult
    │           (ToolResult is a Pydantic model: found/not_found/malformed,
    │            containing only allowlisted fields — see security-model.md)
    │
    ├── if policy/product question:
    │     rag.retrieval.search(query, session_topic_hint) → candidates
    │     rag.authority.gate(candidates) → authoritative vs excluded, with reasons
    │     rag.conflict.check(authoritative) → conflict object or None
    │     rag.evidence.assemble(authoritative, excluded, conflict) → EvidenceBundle
    │
    ▼
llm.prompts.build(system_instructions, EvidenceBundle | ToolResult,
                   bounded_history, current_message)
    │        (untrusted content is wrapped in delimited, labeled blocks;
    │         system instructions are a separate, earlier prompt segment)
    ▼
llm.client.generate() → draft answer (with the model's own claimed sources)
    │
    ▼
validation.response_validator.check(draft, EvidenceBundle | ToolResult, handoff_rules)
    ├─ forbidden-field leakage scan (belt-and-suspenders on top of sanitizer)
    ├─ claimed sources ⊆ authoritative evidence sources
    ├─ conflict present in bundle → both sources cited + handoff true
    ├─ completed-action-claim scan (refund/cancel/replace/address-change)
    └─ if any check fails → bounded single regenerate-with-feedback retry,
       else deterministic fallback ("I can't confirm that — here's what I
       can tell you..." + handoff=true)
    │
    ▼
handoff.decide() merges deterministic rule table (status==exception,
conflict present, evidence empty, unsupported action requested, privacy/
security request) with the validator's outcome → final handoff: bool
    │
    ▼
observability.trace.record(turn)   (redacted, structured, always written)
    │
    ▼
session.update_with_turn_result()  (last_order_id, last_topic_tags, etc.)
    │
    ▼
ResponseEnvelope{ answer, sources[], handoff, tool_calls[], errors[] }
    → interface renders it
```

## 4. Authority / precedence model

Every retrieved chunk carries: `document_id`, `filename`, `heading`,
`status`, `policy_authority`, `audience`, `effective_date`,
`supersedes`, `superseded_by`, plus a retrieval `score` (see §6) and a
curated `topic_tags: list[str]`.

Precedence is evaluated as an **ordered pipeline of gates**, not a single
weighted score, so each decision is independently testable:

1. **Supersession gate.** If a chunk's document has `status: superseded`
   and its `superseded_by` id is present among the active docs, exclude
   it from the *authoritative* set for current-state questions
   (`excluded_reason: "superseded"`). It remains visible in the debug
   trace as a retrieved-but-excluded candidate. Exception: if the user's
   message explicitly asks about historical/prior policy, it may be
   surfaced but labeled "no longer in effect" and never as the answer to
   a current-state question — this exception is a Phase-2 judgment call,
   flagged in `implementation-plan.md`.
2. **Authority gate.** If `policy_authority != "official"` or
   `status not in {"active", "superseded"}` (i.e. `draft`, or any future
   non-approved status), exclude from the authoritative set
   (`excluded_reason: "non_authoritative"`). This is what keeps doc 14
   out regardless of how well it matches the query.
3. **Audience gate.** If `audience == "internal"`, exclude from the set
   of chunks that may ever be *cited to the customer*
   (`excluded_reason: "internal_audience"`). Internal-audience content
   (doc 13) is not retrieved into the customer-answer path at all in the
   default design — its rules are compiled into `handoff.py` at build
   time (see §7) rather than retrieved per-turn, which also means it can
   never accidentally leak as a citation.
4. **Remaining candidates** = the authoritative set. Rank by hybrid
   retrieval score for the purpose of selecting the top-N to include in
   the prompt (N small, e.g. 3–4), but ranking never overrides gates 1–3.
5. **Explicit non-newest-wins rule.** `effective_date` is *never* used to
   break a tie between two members of the authoritative set (per doc
   13's explicit instruction). It is used only as a sanity check that a
   document's `supersedes` claim points to an earlier `effective_date`
   (a data-quality assertion, not a ranking signal).
6. **Conflict check** (§5) runs over the authoritative set before
   evidence assembly.

This directly implements "prefer authoritative, active policy documents
over superseded or non-policy documents" and "do not assume newest
effective_date always wins" as literal, testable code paths.

## 5. Conflict detection

Generalized contradiction detection over arbitrary free text is out of
scope for a 6–8 hour build and would be unreliable. The pragmatic design:

- At ingestion time, each chunk is assigned zero or more `topic_tags`
  via a small deterministic keyword/heading mapping (e.g. a chunk whose
  heading or body matches `/breeze tumbler/i` and `/dishwasher|hand.?wash/i`
  gets tag `breeze-tumbler-cleaning`).
- A **conflict** is flagged when the authoritative set (post-gates)
  contains ≥2 chunks that (a) share a `topic_tags` entry, (b) come from
  different `document_id`s, (c) have no supersession relationship to
  each other, and (d) a small curated extraction (regex over known
  phrases like "dishwasher safe" vs "hand-wash") yields different
  answers for the shared topic.
- This is intentionally a **curated, narrow mechanism** tuned to the
  known corpus (currently one real pair: doc 11 vs doc 12), not a
  general NLP contradiction detector. This tradeoff is called out
  explicitly in `implementation-plan.md` §Ambiguities as something an
  architect should sign off on, since it will not automatically catch
  a *new* kind of conflict introduced by a future document without a
  corresponding tag/extraction rule being added.
- When a conflict is detected, the evidence bundle carries a
  `ConflictObject{ topic, chunk_a, chunk_b }`, and both the prompt
  (as a deterministic system note, not left for the LLM to notice
  unaided) and the response validator require both sources to be cited
  and `handoff=true`.

## 6. Retrieval / indexing approach

- **Chunking:** split by H2 heading within each document; each chunk =
  heading text + its body, tagged with the full parsed front matter of
  its source file and the filename.
- **Index:** a small **local hybrid index** —
  - Lexical: BM25 (`rank_bm25`, pure Python, no model download) — policy
    text is keyword/number-dense ("30 calendar days", "$6.95"), so exact
    lexical matching is a strong, fully deterministic signal and a good
    safety net for the numeric `must_include` assertions in the eval set.
  - Vector: a small local embedding model (e.g. a compact ONNX
    sentence-embedding model via `fastembed`, or `sentence-transformers`
    if the former proves unavailable in the build environment) with
    cosine similarity over an in-process flat NumPy array — appropriate
    at this corpus scale (14 docs, on the order of 40–60 chunks); no
    hosted vector DB is needed and none is used.
  - Combined score = weighted sum of normalized BM25 and cosine scores;
    weights are a config value, tuned against the visible eval set
    during Phase 2.
  - The exact embedding library is a Phase-2 setup decision (flagged in
    `implementation-plan.md`): if installing a local embedding model
    turns out to cost meaningful setup time within the timebox, the
    fallback is BM25-only, which is still a legitimate "local index" and
    keeps the eval suite fully offline/deterministic — the authority and
    conflict layers are unaffected either way since they operate on
    metadata, not on the retrieval score.
- **Top-K:** retrieve K=8 raw candidates, pass through the authority
  pipeline, then include the top N≈3–4 authoritative chunks (plus any
  conflict pair, which is always included together) in the prompt.

## 7. Deterministic vs. retrieved policy knowledge

Two different documents both describe "how the agent should behave"
rather than "what to tell the customer":
- Doc 13 (`support-escalation.md`, `audience: internal`) — handoff
  triggers and the non-newest-wins conflict rule.
- Embedded behavioral sentences inside otherwise customer-facing docs
  (e.g. doc 04/05/07/10's "must not promise/claim X").

Design choice: **compile both into code** (`handoff.py`'s rule table and
the system prompt's fixed guardrail section) at build time, rather than
relying on these being retrieved at the right moment for the right
query. Rationale: these are exactly the "business-critical rules" the
overall brief asks to keep deterministic, they are small and enumerable
(not a large corpus to search), and retrieval-dependent enforcement
would be a reliability regression (a rule only applies if it happens to
be in the top-K for that specific phrasing). Doc 13 is still parsed and
kept in the ingestion index for observability/debug-trace completeness
and because "index the supplied documents" literally includes it — but
it is flagged `citable_to_customer: false` and excluded from every
customer-facing `sources[]` list.

## 8. Order repository & sanitization

- `OrderRepository` loads `data/orders.json` once at startup into an
  in-memory dict keyed by `order_id` (12 records — no database needed at
  this scale; a real system would swap this for a real order service
  behind the same interface).
- `normalize.classify(raw_input) -> NormalizedOrderId | NotAnOrderId |
  None` (None = no order-like token found at all, i.e. the "missing ID"
  case) implements the 3-way classification from `corpus-analysis.md` §4.
- `OrderRecord` (internal Pydantic model, mirrors the full JSON schema
  including `customer` and `internal`) never leaves `orders/` module
  boundaries.
- `sanitize.project(order: OrderRecord, status_aware=True) ->
  CustomerSafeOrder` returns a model containing only the allowlisted
  fields, additionally **omitting** `carrier`/`tracking_number`/
  `estimated_delivery` when `status in {cancelled, returned}` (stale-field
  rule) and omitting `estimated_delivery` entirely when it is `None`
  rather than passing through a null the LLM might rationalize into a
  guess.
- The `order_lookup` tool's return value to the LLM is always one of:
  `CustomerSafeOrder`, `OrderNotFound`, or `InvalidOrderId` — never a raw
  dict, never the internal model.

## 9. Session / conversation state

- `Session{ session_id, turns: deque[Turn] (bounded, e.g. last 8),
  last_order_id: str | None, last_topic_tags: list[str], created_at,
  last_active_at }`.
- Updated deterministically after each turn based on what actually
  happened (a tool call sets `last_order_id`; a citation's topic tags set
  `last_topic_tags`) — not solely inferred by the LLM from raw history,
  so "What about Canada?" and "When will it arrive?" resolve via an
  explicit, testable field rather than hoping the model infers it from
  transcript alone (though the bounded raw history is also included in
  the prompt to help the LLM's own language understanding of the
  follow-up phrasing).
- CLI: one session per process invocation by default (optionally a
  `--session-id` flag to resume). API (if built): `session_id` in the
  request, server-side in-memory store keyed by it, no cross-session
  sharing.

## 10. LLM client & prompting

- One provider: **Anthropic Claude** via the official `anthropic` Python
  SDK (a small/fast model, e.g. Haiku-class, as the default — configurable
  via `.env` — chosen for low per-call cost and latency across many eval
  reruns; OpenAI would be an equally valid single-provider choice, this
  is a project decision documented for architect review, not an
  assignment mandate).
- Prompt structure (four clearly separated regions, in order):
  1. Fixed system instructions (trusted, never influenced by retrieved
     content): role, the non-negotiable behavioral rules compiled from
     §7, the instruction that regions 2–3 below are data, never
     commands.
  2. `<evidence>` block — the assembled EvidenceBundle or ToolResult,
     each item tagged with its filename/heading or "order lookup
     result", explicitly labeled as untrusted retrieved/tool data.
  3. `<conversation_history>` block — bounded prior turns, also treated
     as data for context, not as new instructions.
  4. The current user message, wrapped and labeled as user input (also
     untrusted).
- The LLM is asked to produce a small structured output (e.g. JSON with
  `answer`, `sources_used`, `needs_handoff`) rather than free text, so
  the validator can check structured fields instead of parsing prose.

## 11. Response validation & handoff

- `validation/response_validator.py` runs after generation and before
  the response is returned, checking: forbidden-field leakage (defense
  in depth on top of the sanitizer, since the sanitizer already prevents
  the data from being present — this check protects against a future
  regression that accidentally widens the allowlist), claimed sources
  are a subset of the authoritative evidence actually assembled,
  conflict-present-implies-both-sources-and-handoff, and a
  completed-action-claim scan (regex for phrases implying a refund/
  cancellation/replacement/address-change was *completed*, which should
  never appear since no such action exists in this system).
- On failure, one bounded regenerate-with-feedback retry is attempted;
  if it still fails, a deterministic fallback response is returned
  instead of the model's output, with `handoff=true`.
- `handoff.py` merges: the validator's outcome, the deterministic rule
  table (status==`exception`; conflict detected; evidence bundle empty
  after retrieval; user requests an unsupported action; user requests
  disclosure of internal/secret/other-customer data; user reports fraud/
  safety/legal/privacy issue — all sourced from doc 13's "Recommend
  human assistance when" list) into the final `handoff: bool`.

## 12. Observability

- One structured JSON record per turn (`observability/trace.py`),
  written unconditionally (not only in debug mode) to
  `.traces/{session_id}.jsonl`; a `--debug` CLI flag / `debug=true` API
  param additionally prints it.
- Schema: `session_id, turn_index, user_message, history_window,
  retrieved_chunks[{doc_id, filename, heading, status, policy_authority,
  score, included, excluded_reason}], authority_decisions, conflict,
  tool_calls[{name, args, sanitized_result}], final_response, handoff,
  errors[]`.
- The same allowlist/redaction function used by `orders/sanitize.py` is
  reused here so there is exactly one place that knows which fields are
  forbidden — never a second hand-maintained list that can drift.

## 13. Technology choices (summary)

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Matches the recommended stack; strong typing via Pydantic |
| Interface | CLI (primary), optional thin FastAPI endpoint if time remains | Rubric explicitly does not score visual polish; CLI is fastest to build and to record for the demo GIF |
| Domain models | Pydantic v2 | Typed, validated boundaries everywhere untrusted data crosses into the system |
| Config | `pydantic-settings` + `.env` | Matches `.env.example` requirement, no framework overhead |
| Lexical retrieval | `rank_bm25` | Pure Python, zero model download, deterministic |
| Vector retrieval | Small local embedding model (`fastembed` or `sentence-transformers`) + flat NumPy cosine index | "Local vector index" without a hosted vector DB dependency; corpus is far too small to need FAISS/Chroma-scale infra |
| LLM provider | Anthropic Claude (`anthropic` SDK), one model family | Single provider as instructed; small/fast model keeps eval reruns cheap |
| Testing | `pytest` | Standard, matches recommended stack |
| Logging | Stdlib `logging` with a JSON formatter (or `structlog` if it saves real time) | "Plain structured logs are sufficient" — no dashboard |
| CI | GitHub Actions | Runs `pytest` + the evaluation suite on every push, matches recommended stack |

## 14. Repository structure (proposed for Phase 2)

```
.
├── README.md
├── .env.example
├── pyproject.toml
├── src/aster_row_agent/
│   ├── config.py
│   ├── models.py
│   ├── interface/{cli.py, api.py}
│   ├── orchestration/{agent.py, session.py}
│   ├── rag/{ingest.py, index.py, retrieval.py, authority.py, conflict.py, evidence.py}
│   ├── orders/{repository.py, normalize.py, sanitize.py, tool.py}
│   ├── llm/{client.py, prompts.py}
│   ├── validation/response_validator.py
│   ├── handoff.py
│   └── observability/trace.py
├── knowledge-base/        (copied verbatim from the assignment repo)
├── data/                  (copied verbatim from the assignment repo)
├── evaluation/
│   ├── visible-cases.json (copied verbatim)
│   ├── custom-cases.json  (≥5 original cases)
│   ├── runner.py
│   └── results/{baseline.json, final.json}
├── tests/{unit/, integration/}
├── docs/                  (this Phase 1 output)
└── .github/workflows/ci.yml
```

`knowledge-base/` and `data/` are copied as-is from the assignment
repository (not modified, not regenerated) so the shipped submission is
self-contained and reproducible from a clean clone without depending on
`reference/ai-agent-intern-test` remaining available.

## 15. Phase 2 as-built: ingestion and the evidence foundation

This section records what was actually implemented for the RAG ingestion
pipeline (`src/aster_row_agent/rag/`), superseding §6 and §14 above
wherever they differ.

**Ingestion architecture.** Ingestion is a one-way pipeline with no
knowledge of retrieval, authority, or conversation state:
`discover_documents` → `parse_corpus` → `build_document_chunks` (per
document) → `embedding_provider.embed_documents` → `save_index` +
`write_manifest`. `rag/ingest.py` is the only module that sequences these
steps; every other `rag/*` module is independently unit-testable and has
no dependency on `ingest.py`. Retrieval/authority/conflict-detection
(originally §4-5 above) are deliberately not implemented yet — Phase 2
stops at "evidence," per the phase brief.

**Metadata model.** `DocumentMetadata` (`rag/models.py`) requires only the
five fields present on every one of the 14 supplied documents
(`document_id`, `title`, `status`, `audience`, `policy_authority`);
everything else observed in the corpus (`effective_date`, `last_reviewed`,
`supersedes`, `superseded_by`, `superseded_date`, `customer_answering`) is
optional. `model_config = ConfigDict(extra="allow")` means a front-matter
key this model doesn't yet know about is preserved on the object (visible
via `.model_extra` and included in `model_dump()`) rather than silently
dropped. The model is `frozen=True` — metadata cannot be mutated in
memory after parsing.

**Chunking strategy.** Implemented in `rag/chunking.py` exactly as
designed: `parse_heading_sections` walks the Markdown body with a level
stack (handles arbitrary heading depth, not just H1/H2, even though the
real corpus only uses two levels) and attributes each span of body text to
its full heading path; `split_section_text` only sub-splits a section when
it exceeds `max_chunk_chars` (default 1200), packing at paragraph
boundaries and hard-splitting a single oversized paragraph only as a last
resort. Against the real 14-document corpus this produces exactly one
chunk per heading section (53 chunks total, 2-5 per document) — no real
section is anywhere near the size threshold, confirmed by running
ingestion. The sub-splitting path is proven correct with a synthetic
fixture in `tests/unit/test_chunking.py`, since the real corpus cannot
exercise it.

**Provenance & citation.** Every `DocumentChunk` carries its full
`DocumentMetadata`, `source_path`, `title`, `heading_path`, and
`chunk_index` — a citation string (`"<filename> — <heading path>"`) is
derivable from the chunk alone via its `.citation` property, with no
dependency on an LLM having remembered where the text came from. Example,
verified in `tests/unit/test_ingest.py`:
`"01-returns-policy-current.md — Returns Policy > Standard return window"`.

**Deterministic IDs & content hashing.** `rag/hashing.py` provides
`hash_bytes`/`hash_text` (SHA-256) and `compute_chunk_id`, which hashes
`document_id || source_path || heading_path || chunk_index ||
content_hash` (truncated to 24 hex chars). No UUIDs are used anywhere in
the pipeline. A document-level `content_hash` (over the full raw file
bytes, front matter included) feeds a corpus-level `corpus_fingerprint` in
the manifest — a single hash that changes if and only if some document's
bytes changed.

**Index strategy.** As flagged for review in `implementation-plan.md`
§3.2, the local embedding dependency was evaluated directly rather than
assumed: `fastembed` (ONNX runtime, no torch) installs in ~25s and
`BAAI/bge-small-en-v1.5` (384-dim, ~67MB) downloads and embeds the whole
corpus in under a second once cached. **No fallback to BM25-only was
needed** — the real local embedding model is the shipped default. BM25
and the retrieval/authority fusion logic described in §4-6 above remain
deferred to the next phase, since they are query-time concerns, not
ingestion-time artifacts. The index itself (`rag/index.py`) is a flat
JSON-Lines file (`records.jsonl`, one self-describing `IndexRecord` per
line — chunk + embedding + model id) plus `manifest.json`, written
atomically (temp file + rename) as a full rebuild on every ingestion run —
never an incremental append. `VectorIndex.search()` performs cosine
similarity over an in-memory NumPy matrix; manually verified to correctly
surface both sides of the doc 11/12 Breeze Tumbler conflict near the top
of the same query's results (see Phase 2 completion report), confirming
the evidence layer will give the next phase's conflict detector what it
needs.

**Embedding abstraction.** `rag/embeddings.py` defines an
`EmbeddingProvider` protocol (`embed_documents`, `embed_query`, `model_id`,
`dimensions`). `FastEmbedProvider` is the real provider; a
`FakeDeterministicEmbeddingProvider` (pure SHA-256-derived vectors, no
model, no network) is used by every unit test and is available via
`--fake-embeddings` on the CLI for fast/offline smoke runs. This is why
the full test suite runs in well under a second with zero network calls.

**Reproducibility.** Re-running `python -m aster_row_agent.cli ingest`
against an unchanged corpus regenerates byte-identical chunk IDs and an
identical `corpus_fingerprint` (verified in
`test_run_ingestion_is_idempotent_over_unchanged_corpus`), because every
identifier is a pure function of content, never a timestamp or random
value, and `save_index`/`write_manifest` always write a full, atomic
snapshot rather than appending.

**Testing strategy.** 85 tests, all deterministic, none requiring network
access, an LLM, or the real embedding model (`FakeDeterministicEmbeddingProvider`
is used throughout `tests/`, though the real `FastEmbedProvider` path was
also manually exercised end-to-end — see the Phase 2 completion report).
Coverage includes: front-matter edge cases (missing/malformed/empty),
metadata tolerance for optional and unknown fields, heading-parsing edge
cases (nesting, preamble, empty sections), chunk-splitting edge cases
(oversized sections, oversized single paragraphs), deterministic ID/hash
properties, index save/load/search roundtrips, full end-to-end ingestion
over the real 14-document corpus, idempotency, byte-for-byte source
immutability, metadata preservation for the superseded/internal/draft
documents specifically, and verbatim (inert) preservation of the embedded
prompt-injection text. No test hardcodes an answer to any
`evaluation/visible-cases.json` case.

## 16. Phase 3 as-built: retrieval, authority, conflict, evidence

This section records what was actually implemented for the retrieval/
evidence decision layer (`src/aster_row_agent/rag/{query,lexical,
retrieval,authority,applicability,claims,conflict,evidence,
retrieval_models}.py`), refining §4-6 above based on explicit Phase 3
guidance and empirical measurement against the real corpus and the real
embedding model.

**One deviation from §4's original design, made deliberately.** §4.3
originally said internal-audience content (doc 13) would not be retrieved
into the customer-answer path at all. The Phase 3 brief was explicit that
this was too aggressive: Stage A (retrieval) must retrieve from the
*entire* candidate pool — including internal and non-authoritative
chunks — because excluding them at retrieval time would prevent
legitimate uses (an internal escalation chunk informing a handoff
decision; a non-authoritative chunk needing to be visible in diagnostics
for exactly the prompt-injection/fabricated-policy scenarios the corpus
tests). As implemented: **no chunk is ever excluded from retrieval** based
on its metadata. Authority and citability are decided only afterward, per
chunk, from metadata alone (`rag/authority.py`), and only citability
(never retrieval, never relevance) gates what a customer answer may cite.
Doc 13's rules are *also* still intended to inform a deterministic
handoff engine at build time in a later phase (§7 is otherwise
unchanged) — the two are complementary, not exclusive.

**Hybrid retrieval and score fusion.** `rag/lexical.py` wraps `rank_bm25`
over a stopword-filtered tokenizer (removing stopwords was not in the
original plan — see "risks discovered" below); `rag/index.py`'s
`VectorIndex` (from Phase 2) gained a `score_all()` method alongside its
existing `search()`, so the same index serves both a simple top-k lookup
and the exhaustive per-query scoring the hybrid retriever needs.
`rag/retrieval.py`'s `Retriever` scores *every* chunk with both methods
on every query (cheap at this corpus's scale — ~53 chunks), independently
min-max normalizes each score list to [0, 1] across the current pool, and
fuses them with configurable weights (default 0.5/0.5). Every candidate
carries all four numbers (`lexical_score`, `semantic_score`, and their
normalized counterparts) plus `fused_score`, so the exact contribution of
each method is inspectable per query, per chunk — never collapsed into an
opaque single number.

**Query normalization** (`rag/query.py`) is exactly as conservative as
specified: Unicode NFKC + whitespace collapsing, nothing else. Case
normalization is left to each retrieval method internally (BM25's
tokenizer lowercases; embeddings are case-insensitive by construction)
rather than applied to the shared normalized string, so a future
consumer that needs original casing (e.g. an order-ID extractor) is not
affected by this layer.

**Authority model** (`rag/authority.py`). `classify_chunk_authority` is a
4-line priority pipeline over `status`/`policy_authority`/`audience`
alone, producing one of `AUTHORITATIVE_CUSTOMER`, `SUPERSEDED`,
`NON_AUTHORITATIVE`, `INTERNAL_OPERATIONAL`, or (a data-quality fallback
never actually produced by this corpus) `UNRECOGNIZED_AUDIENCE`. Checking
"not active/superseded, or not officially authored" *before* the internal
check is what correctly separates the migration scratchpad
(`NON_AUTHORITATIVE` — it's draft *and* internal) from the legitimate
support-escalation policy (`INTERNAL_OPERATIONAL` — active, official,
merely internal-audience); collapsing these into one bucket would have
been a real correctness bug. `analyze_supersession` cross-checks a
document's supersession claim against the loaded corpus (does the
superseding document exist? is it active?) and — the one place
`effective_date` participates in authority evaluation at all — flags
`date_ordering_consistent=False` if a document's claimed successor has an
*earlier* effective date than itself, a pure data-quality sanity check
never used for ranking or tie-breaking (INVARIANT 3).
`tests/unit/test_authority.py::test_no_filename_based_authority_logic`
and `::test_no_filename_based_conflict_logic` read the actual module
source and fail if any real corpus filename or document_id ever appears
in it — a standing, automated guard for INVARIANT 2, not just a design
promise in prose.

**Conflict detection** (`rag/claims.py` + `rag/conflict.py`) implements
the hybrid strategy the brief requires — not a filename registry. Two
independent, narrow `ClaimExtractor`s (`return_window_days`,
`breeze_tumbler_body_dishwasher_safe`) pull structured, directly
comparable values out of individual sentences via regex, gated on
context words ("return", "tumbler") rather than section identity, so
they generalize to *any* chunk containing matching language regardless
of which document it's in. `rag/applicability.py` tags each claim's
source sentence with any named segment it mentions (`trailplus`,
`standard`, `final_sale`, `domestic`, `international`, `canada`); two
claims on the same concept are only compared if their tag sets are not
*both non-empty and disjoint* — this is what correctly keeps the
standard-30-day and TrailPlus-45-day windows from being flagged as
conflicting with each other, while still catching the fabricated
60-day claim (which, once a "every item"/"every customer" universal-
scope override is applied — see below — is correctly treated as
unscoped and therefore comparable). `rag/conflict.py` then classifies
every same-concept, cross-document, differing-value pair through
`_classify_disagreement`, which consults each side's independently-
computed `AuthorityDisposition` to produce one of
`GENUINE_ACTIVE_CONFLICT`, `RESOLVED_BY_SUPERSESSION`,
`NON_AUTHORITATIVE_DISAGREEMENT`, `INTERNAL_OPERATIONAL_DISAGREEMENT`, or
`UNCERTAIN` (never fabricated when no rule cleanly applies). A secondary,
purely diagnostic topic-keyword-overlap signal
(`shared_topic_keywords`) is attached to every detected `Conflict` for
transparency but is **not** a gate — precise concept-key matching is
already the stronger signal (see `rag/conflict.py`'s module docstring for
the specific case — doc 01 vs. doc 14 — that a topic-overlap gate would
have incorrectly suppressed).

Empirically verified against the real corpus (not just synthetic
fixtures): `detect_conflicts` over all 14 real documents produces exactly
one `GENUINE_ACTIVE_CONFLICT` (doc 11 vs. doc 12, as designed), one
`RESOLVED_BY_SUPERSESSION` (doc 01 vs. doc 02), and three
`NON_AUTHORITATIVE_DISAGREEMENT`s (doc 14 vs. each of doc 01/02/09) — see
`tests/unit/test_conflict.py::test_detect_conflicts_over_real_corpus_finds_expected_dispositions`.

**Evidence assembly** (`rag/evidence.py`) implements the exact Stage
A→B→C pipeline the brief specifies. Sufficiency is decided in **two
stages using two different score spaces** — this was a real bug caught
during calibration (see "risks discovered" below) and is now the load-
bearing design point of the whole layer: an **absolute** floor
(`min_semantic_score=0.60`, `min_lexical_score=2.0`, combined with OR)
checked against each candidate's *raw*, un-normalized score, followed by
a **relative** margin (`relevance_margin=0.5`) applied only among
already-absolute-qualified candidates using the fused, pool-normalized
score. `select_evidence` always includes both sides of a genuine conflict
together when both are relevant (never one side silently); otherwise
fills up to `max_selected_evidence` (default 4) with
`AUTHORITATIVE_CUSTOMER` candidates by score; if none exist, lets exactly
one `INTERNAL_OPERATIONAL` candidate through for handoff context, never
as a citable source. Conflicts reported on the bundle are filtered to
only those whose both sides are relevant *to this query* — an early
version leaked an unrelated latent conflict (Breeze Tumbler) into an
"international shipping" query's debug output simply because both
tumbler chunks happened to land in the raw top-8; this is now excluded
(§15, topic contamination) without weakening `rag/conflict.py`'s own
corpus-wide detection.

**Technology.** BM25 via `rank_bm25`; the real local embedding model from
Phase 2 (`BAAI/bge-small-en-v1.5` via `fastembed`) — no fallback to
BM25-only was needed, contrary to the fallback Phase 1/2 kept available
as a hedge. No new dependency was added.

### Empirical calibration data (referenced by `EvidenceAssemblyOptions`)

Raw cosine similarity (same embedding model, same corpus), top-3 hits per
query, measured directly rather than assumed:

| Query class | Example | Raw semantic score range |
|---|---|---|
| On-topic (exact terms) | "How long do I have to return an item?" | 0.76 - 0.80 |
| On-topic (country) | "Do you ship to Canada?" | 0.66 - 0.77 |
| On-topic (heavy paraphrase, ~zero lexical overlap) | "if I dont like the color of my bag can i send it back" | 0.64 - 0.69 |
| Off-topic | weather / sports / recipes / taxes / dog leashes / coding | 0.44 - 0.59 |

The gap between the worst on-topic score (0.64) and the best off-topic
score (0.59) is where `min_semantic_score=0.60` was placed. This is a
starting point for further empirical tuning during the evaluation phase
(docs/evaluation-plan.md), not a final calibrated value.

### Risks discovered during Phase 3 (see also the final response's "risks" item)

1. **Min-max normalization cannot detect "nothing is relevant."** Because
   min-max normalization always stretches the current pool's best score
   to 1.0 by construction, an early implementation that checked
   sufficiency against the *fused, normalized* score could never produce
   `INSUFFICIENT_EVIDENCE` — even a pool of uniformly irrelevant chunks
   has *a* best-scoring member, and it would always normalize to the
   maximum. Caught empirically (a "What's the weather today?" query
   scored a suspicious, suspiciously round `0.5`), and fixed by moving
   the absolute floor onto raw, un-normalized scores (see above). This is
   exactly the kind of design flaw a spec review alone would not have
   surfaced — it only showed up by actually running real queries through
   the real pipeline.
2. **BM25 needs stopword filtering at this corpus's scale.** Without it,
   a common function word ("how", "does", "what") that happens to be
   locally rare across ~53 short chunks could receive a pathologically
   high IDF weight and outrank a genuinely on-topic document — observed
   directly: an internal support-escalation chunk briefly outranked the
   actual Canada-shipping document for "What about Canada, and how long
   does it take?". Fixed with a standard small English stopword list in
   `rag/lexical.py`.
3. **Retrieval relevance is not the same as answer sufficiency, and this
   layer cannot fully close that gap.** For a query like "Are all fabrics
   and adhesives in your bags vegan?", the retriever correctly finds
   genuinely on-topic, authoritative evidence (the bags/backpacks care
   instructions) with a high relevance score — but that evidence does not
   actually address veganism at all. This layer reports `ANSWERABLE`
   because real, relevant, authoritative content exists; recognizing that
   the content doesn't address the *specific* question asked requires
   reading comprehension this evidence layer does not attempt, by
   design (see docs/architecture.md §1 principle 1 and the module
   docstring in `rag/evidence.py`). This is flagged, not silently
   accepted, as the primary remaining architectural risk for the next
   phase — see the Phase 3 completion report's "risks" section for the
   recommended mitigation (a response-validation step that checks the
   generated answer's claims against the evidence text, not just the
   evidence bundle's disposition).

## 17. Phase 4 as-built: order lookup and the secure order boundary

New package: `src/aster_row_agent/orders/{models,errors,normalize,sanitize,
repository,service}.py`. `data/orders.json` and
`data/orders-data-dictionary.md` were copied verbatim into the project
(same pattern as `knowledge-base/` in Phase 2). `Settings` gained one new
field, `orders_file` (default `data/orders.json`) — no other existing
module was changed except `config.py`.

**The boundary, exactly as required:**

```
orders.json -> OrderRepository -> RawOrderRecord
                                        |
                                   sanitize_order()   <- the ONLY place
                                        |                 Raw -> Safe happens
                                        v
                                 CustomerSafeOrder
                                        |
                              OrderLookupService.lookup()   <- the tool-facing
                                        |                       contract
                                        v
                               OrderLookupResult (typed, JSON-serializable)
```

**Raw schema** (`RawCustomerInfo`, `RawOrderItem`, `RawInternalInfo`,
`RawOrderRecord`) mirrors `orders.json` field-for-field, including
`customer.*` and `internal.*`. All four models use `extra="allow"` —
this is a deliberate, Phase-2-precedented choice (matches
`DocumentMetadata`'s design): a future field added to the dataset is
captured, not silently discarded, so the repository doesn't crash on
schema evolution, but that captured data is never automatically
exposed — only fields the sanitizer explicitly names ever reach
`CustomerSafeOrder`. Verified directly:
`test_unknown_future_field_on_raw_record_does_not_leak` constructs a
raw record with two invented sensitive-looking fields
(`customer_date_of_birth`, `loyalty_program_ssn`), confirms
`RawOrderRecord` captures them in `model_extra`, and confirms neither
appears anywhere in the sanitized output.

**Customer-safe schema** (`CustomerSafeOrder`, `CustomerSafeOrderItem`)
is the allowlist from `data/orders-data-dictionary.md`, verbatim, plus
two derived fields computed from already-safe data:
`requires_support_review` (`status == "exception"`) and
`stale_delivery_fields_suppressed` (whether the stale-field rule below
fired) — added so a future agent reads a typed boolean instead of
re-deriving business rules from a raw status string. `extra="forbid"`
on both models means attempting to construct one with an unexpected
field (e.g. `email=...`) is a `pydantic.ValidationError`, not a silent
pass-through — the strict side of the boundary. `items.sku` is
deliberately excluded: it is not in the data dictionary's customer-safe
list.
`tests/unit/test_order_models.py::test_customer_safe_order_field_set_is_exactly_the_documented_allowlist`
pins the exact field set so any future addition must consciously update
that test.

**Repository** (`OrderRepository.load`) is fail-fast, once, at
construction: invalid JSON, wrong top-level shape, any invalid record
(pydantic validation, collected across *all* records before raising —
same pattern as `rag.ingest.parse_corpus`), and duplicate `order_id`s
each raise a specific typed error and prevent the repository from being
constructed at all. There is no partial/best-effort repository and no
silent pick-one-of-duplicates. Once constructed, `get_by_order_id` is a
plain dict lookup — no reread of the file, no per-query parsing.

**Order-ID normalization** (`normalize_order_id`) accepts only harmless
formatting variance (case, surrounding whitespace, the separator between
"ORD" and the digits) via one anchored regex
(`^ORD[\s\-_]*([0-9]{4,10})$`). Because the pattern is an anchored
allowlist, adversarial input (paths, shell/SQL fragments, injected
"instructions", oversized strings, Unicode lookalike digits, RTL
override characters) simply fails to match — no special-casing was
needed, and `tests/unit/test_order_normalize.py` verifies this directly
against ~15 adversarial inputs. Per the assignment's explicit
instruction, `"1007"` is never guessed to mean `"ORD-1007"` — an input
that doesn't already look like an order ID is `MALFORMED`, not
fuzzy-matched.

**Lookup result contract** (`OrderLookupOutcome`): `FOUND`,
`MISSING_ORDER_ID`, `MALFORMED_ORDER_ID`, `NOT_FOUND`, `DATASET_ERROR` —
five states a future agent branches on explicitly, never infers from a
`None` field. `DATASET_ERROR` is a real, tested path
(`OrderLookupInfrastructureError`, raised by a fake repository in
`test_order_service.py`) even though today's in-memory dict-backed
repository can never actually trigger it — the contract exists now for
a future repository implementation (e.g. networked) that could.

**Status authority and stale-field suppression** (`sanitize_order`):
`status` passes through unchanged and is never inferred from
`carrier`/`tracking_number`/presence of a shipped/delivered timestamp.
When `status in {"cancelled", "returned"}`, `carrier`, `tracking_number`,
and `estimated_delivery` are set to `None` in the safe projection
regardless of what the raw record contains — verified against the real
`ORD-1004` (cancelled) and `ORD-1008` (returned) records, whose raw data
genuinely does retain stale carrier/tracking/ETA values. `placed_at`/
`shipped_at`/`delivered_at` are *not* suppressed — they are historical
facts, not stale projections, and the data dictionary calls out only
carrier/tracking/ETA specifically.

**ETA handling**: `estimated_delivery` is passed through as-is when
present, and stays `None` when the raw value is `None` — there is no
code path anywhere in `sanitize_order` that computes, derives, or
defaults a date. Verified against the real `ORD-1011` (shipped, ETA
genuinely unavailable — carrier's feed is down) and `ORD-1010`
(exception, ETA null) records.

**Observability**: `order.lookup.started`/`order.lookup.completed`
events, with fields limited to `canonical_order_id`, `outcome`,
`duration_ms`, and — only when an order was found —
`status`/`eta_available`/`stale_delivery_fields_suppressed`/
`requires_support_review`, all read off `CustomerSafeOrder`. There is no
code path in `service.py` that has a reference to a `RawOrderRecord` at
logging time — logging cannot leak what it structurally cannot reach.
Verified directly: `test_lookup_logs_do_not_contain_forbidden_fields`
inspects the actual structured `context` dict (not just rendered log
text) for `ORD-1007` (whose raw record contains an email, address, and a
risk score of 82) and confirms none of it is present.

**What this module does not do**, by design (INVARIANT 13 and the
Phase 1 module-boundary split): `OrderLookupService` never answers a
policy question, never decides conversational context, never generates
natural-language text, and never calls an LLM. It returns exactly one
`OrderLookupResult` per call. Wiring this into an actual LLM
tool-calling loop, and deciding what a response says about a `FOUND`/
`NOT_FOUND`/`DATASET_ERROR` result, is Phase 5's job.

## 18. Phase 5 as-built: session, routing, LLM integration, and prompt safety

New package: `src/aster_row_agent/agent/{models,errors,session,routing,
llm,prompts,handoff,validation,orchestration}.py`. Nothing in `rag/` or
`orders/` was modified — this package only *consumes* `EvidenceBundle`
and `OrderLookupResult` as already-frozen, typed contracts. `config.py`
gained three fields (`llm_model`, `llm_max_tokens`,
`llm_timeout_seconds`); `pyproject.toml` gained one dependency
(`anthropic`, imported lazily — no test in the project requires it to be
configured with a real key).

**Turn flow, exactly as specified**: user message → `SessionStore` →
`routing.classify_message` (deterministic) → `EvidenceAssembler`/
`OrderLookupService` (Phases 3-4, untouched) → `handoff.decide_pre_llm_handoff`
→ `prompts.build_user_content` → `LLMClient.generate` → `validation.validate_response`
→ citation rendering → `AgentResponse` + session update + structured
trace. `orchestration.Agent.handle_message` is the single entry point;
every other module is independently unit-tested without it.

**The central design decision**: deterministic short-circuits skip the
LLM entirely wherever the correct response is already fully determined
by typed data — sensitive requests, missing/malformed order IDs, unknown
orders, dataset errors, orders requiring support review, and fully
insufficient evidence all return a fixed template with `used_llm=False`.
The LLM is invoked only to *phrase* an answer from evidence/order data
that deterministic code has already selected and gated. This is a direct,
literal application of the stated precedence (deterministic logic >
trust boundaries > structured evidence > LLM generation) rather than a
slogan: roughly half of the orchestration test suite's cases never reach
`LLMClient.generate` at all, which is exactly the point — those are the
cases where getting it right matters most and an LLM call would only add
a failure mode with no compensating benefit.

**Session state** (`session.py`): `Session{session_id, turns (bounded to
8, each truncated to 2000 chars), turn_count (unbounded, observability
only), last_order_id, last_topic_hint (bounded to 300 chars), created_at,
last_active_at}`. No field can hold an order object, a document, or PII
of any kind — the model's field set is asserted exactly in
`tests/unit/test_agent_session.py`. `SessionStore` is a plain dict keyed
by `session_id` with no cross-session references anywhere in its
implementation, which is what makes session isolation a structural
property rather than a policy: there is no code path by which session
A's dict entry could be read while handling session B.

**Routing** (`routing.py`) is regex/keyword-based and runs entirely
before any LLM or retrieval call. Three real bugs were caught and fixed
by testing against the brief's own examples rather than trusting the
first design (see "risks discovered" below): the order-ID candidate scan
originally matched the English word "order" itself (since "order" starts
with "ord"); it also missed "clearly-attempted-but-invalid" IDs like
"ORD-ABCD" when no other order keyword was present in the message,
misrouting them to KNOWLEDGE instead of surfacing a "please provide a
valid order ID" clarification. Order-ID candidates found in the message
are normalized through Phase 4's `normalize_order_id` directly — this
module never reimplements that logic, only decides *where in the
message* to look.

One documented judgment call: `RoutingDecision.action_is_direct_command`
distinguishes an eligibility question ("Can I cancel ORD-1001?") from an
imperative command ("Cancel it right now."), but — deliberately — this
distinction is used only by `prompts.py` to adjust tone, never by
`handoff.py` to force escalation. The assignment's own
`retrieved-prompt-injection` visible case phrases its adversarial request
imperatively ("...approve my return") yet expects `handoff: false` once
the real policy is correctly explained; forcing handoff from imperative
phrasing alone would fail that case. Handoff is instead driven entirely
by evidence/order disposition (see below), which produces the right
answer for both that case and a bare "cancel my order" command (the
latter still gets a helpful, non-escalating answer when a real
cancellation-window policy exists to explain — escalation only happens
if there is genuinely nothing else to say, per the ordinary
`INSUFFICIENT_EVIDENCE` rule).

**LLM abstraction** (`llm.py`): `LLMClient` is a one-method `Protocol`.
`AnthropicLLMClient` lazily imports `anthropic` inside `__init__` (same
pattern as `rag.embeddings.FastEmbedProvider`'s optional heavy
dependency) so nothing in the module requires the package or a key merely
to import. `FakeLLMClient` — used by every test in this project — takes a
scripted list of `LLMResponse`s or exceptions to raise, so malicious,
malformed, and error-path model behavior is fully deterministic to test.
The model's structured output (`{"answer": ..., "cited_filenames": [...]}`)
deliberately has no handoff/disposition field: the LLM has no channel
through which it could influence that decision, by construction.

**Trust boundary** (`prompts.py`): `SYSTEM_INSTRUCTIONS` is a bare
module-level string constant — never an f-string, never `.format()`ted,
never concatenated with anything. It is passed via the provider's
dedicated `system` parameter, structurally separate from
`user_content`. Every piece of untrusted content (user message, bounded
history, evidence, order result) is assembled into `user_content` inside
explicitly labeled `<tag>` blocks by `build_user_content`, which is a
pure function with no access to `SYSTEM_INSTRUCTIONS` at all — there is
no code path in this module that could let untrusted text reach the
trusted channel. Only `EvidenceBundle.authoritative_evidence` is ever
rendered into the evidence block; `internal_evidence` and
`non_authoritative_evidence` inform `handoff.py`'s decision directly from
Python and are never shown to the model, so there is no way for the LLM
to paraphrase internal-only text into something that reads like a
citation.

**EvidenceBundle / OrderLookupResult integration**: `_render_evidence_block`
and `_render_order_block` (`prompts.py`) read only the already-vetted
subsets (`authoritative_evidence`, `CustomerSafeOrder`) — never
`internal_evidence`, `non_authoritative_evidence`, `RawOrderRecord`, or
any field Phase 4 excluded from the customer-safe allowlist. A stale
suppressed order (cancelled/returned) renders an explicit instruction
("do not mention a delivery estimate or carrier for this order") rather
than simply omitting the fields silently, so the LLM is told *why*, not
left to guess.

**Response validation** (`validation.py`) implements every category the
brief lists: action-completion claims (regex, distinguishing "your order
is cancelled" — a status report — from "I've cancelled your order" — a
completion claim); order claims (stale-delivery-claim and fabricated-ETA
detection, checked against the same `CustomerSafeOrder` fields the prompt
was built from); citations (any filename not in
`customer_citable_sources` is rejected); disposition consistency
(citing anything under `INSUFFICIENT_EVIDENCE`, or citing only one side
of a `GENUINE_ACTIVE_CONFLICT`, both fail); internal-data leakage (email
regex, named-internal-field phrases, system-instruction echo); and the
Phase 3-identified "retrieval relevance != evidence sufficiency"
regression (a small, explicit, extensible list of fact-sensitive terms —
documented as one signal in a disposition-driven architecture, not a
general solution). Any failure discards the LLM's answer entirely and
substitutes a fixed fallback — never a "repaired" version of the model's
text, since reliably repairing free text is itself unsolved.

**Handoff** (`handoff.py`) is a priority-ordered, side-effect-free
function over typed inputs: sensitive request > authoritative conflict >
non-authoritative-only > order dataset error > order not found > order
requires support review > insufficient evidence (only when no order
result compensates). Nothing the LLM produces can change this decision —
`orchestration.py` computes it *before* calling `generate()`, and only
adds `VALIDATION_FAILED`/`LLM_UNAVAILABLE` afterward if generation itself
had a problem.

**Multi-turn behavior**: order-ID continuity is a single deterministic
field (`Session.last_order_id`), set only when the *message itself*
established a well-formed ID — never guessed, never carried across
sessions. Topic continuity uses a conservative two-pass retrieval
strategy (`Agent._maybe_assemble_evidence`): the current message is
tried alone first; only if that comes back `INSUFFICIENT_EVIDENCE` is it
retried once, concatenated with the *immediately preceding* user message
only (never the full history). This means a strong standalone query is
never diluted by stale context, and a genuinely terse follow-up ("What
about Canada?") still gets a chance to resolve via the prior turn — both
properties verified directly in `test_agent_orchestration.py`'s
multi-turn cases, including the case explicitly checking that an
unrelated new topic (an address-change request) never inherits a prior
shipping-policy citation.

**Observability**: `agent.turn.started`/`agent.turn.completed` structured
log events carry `route_kind`, disposition, handoff/reason,
`validation_passed`, `used_llm`, source count, and duration — never a
raw order field or the full prompt text. `agent.validation_failed` logs
only the issue *codes* (e.g. `"stale_delivery_claim"`), never the
offending answer text itself, so a validation failure is diagnosable
without the log line itself becoming a leakage vector.

### Risks discovered during Phase 5

1. **The order-ID routing regex initially matched the English word
   "order" itself**, because "order" starts with the letters "ord" and
   the first draft's pattern accepted any letters following an optional
   separator. Caught by testing the brief's own "Where is my order?"
   example and getting `MALFORMED_ORDER_ID` instead of
   `MISSING_ORDER_ID`. Fixed by requiring either an explicit separator
   (hyphen/underscore/space-then-digits) or digits glued directly to
   "ORD" — a real or attempted ID always has one of these; plain English
   continuations of the word never do.
2. **A malformed-but-attempted ID with no other order keyword was
   misrouted to KNOWLEDGE.** "Check ORD-ABCD please." has no separate
   word like "order" or "tracking" for the intent scanner to key off, so
   `has_order_intent` was initially false and the message fell through to
   a policy-question route, silently skipping the malformed-ID
   clarification entirely. Fixed by treating "an order-ID-shaped
   candidate was found in the message" as order intent on its own,
   independent of whether normalization later succeeds.
3. **The command-vs-question handoff heuristic was reconsidered mid-
   implementation** (see the routing section above) after checking it
   against the assignment's own visible case rather than only against a
   test case invented for this project — a reminder that an internally
   consistent-seeming rule can still be wrong until checked against the
   actual required behavior.
4. **Unresolved, flagged rather than silently decided**: the
   unsupported-claim validator (`_unsupported_claim_issue`) is a curated,
   explicit term list (vegan/organic/sustainable/...), which is the
   documented, intentional scope for this phase — but it will not catch
   an equivalent unsupported claim about a term not on that list. This is
   the same "retrieval relevance != evidence sufficiency" gap Phase 3
   identified as a fundamental limit of a non-LLM-judged architecture,
   narrowed but not closed. A more general fix (e.g. asking the LLM
   itself, as a second, separately-validated call, whether its own answer
   is entailed by the provided evidence) is a reasonable Phase 6+
   candidate but was judged out of proportion for this phase's timebox.

## 19. Phase 6 as-built: CLI application, end-to-end integration, response hardening

Phase 6 adds the one thing every prior phase existed to serve: a real,
interactive entry point that composes the Phase 2-5 layers into a working
agent and talks to a real LLM provider. It intentionally adds no new
business logic — the CLI is a thin composition root and a read-print loop.

### 19.1 CLI surface

Two subcommands under one entry point (`python -m aster_row_agent.cli`):

- `ingest` — unchanged from Phase 2.
- `chat` — new. Starts an interactive session: prints a banner, then loops
  `read a line -> handle_message -> render -> print` until `exit`/`quit`,
  Ctrl+C, or EOF.

`src/aster_row_agent/cli.py` contains no routing, retrieval, sanitization,
prompting, or validation logic of its own — every decision of that kind
happens in the Phase 3-5 layers it calls into.

### 19.2 Application composition (the one path)

`build_agent(settings: Settings) -> Agent` is the single, real composition
path, matching the module map's intended wiring:

```
Settings
  -> FastEmbedProvider -> EvidenceAssembler.load(index_dir, embedding_provider)
  -> OrderRepository.load(orders_file) -> OrderLookupService
  -> AnthropicLLMClient(api_key from environment, model from Settings)
  -> Agent(session_store=SessionStore(), evidence_assembler=..., order_lookup_service=...,
           llm_client=..., model=..., max_tokens=..., timeout_seconds=...)
```

There is no other place in the codebase that constructs a "real" `Agent`.
`SessionStore()` is created exactly once per CLI process, in `build_agent`,
and its single instance is threaded through `run_chat_loop` for every turn
of the session — a new `SessionStore` per turn would silently discard
conversation memory, which is exactly the Phase 5 behavior Phase 6 must
not regress. `ANTHROPIC_API_KEY` is read directly from `os.environ` inside
`build_agent`, and only there; it is checked before any index or dataset
loading so a missing-credential failure is immediate, independent of
whether the knowledge base has been ingested yet, and requires no real
index to test.

### 19.3 Real LLM configuration

`AnthropicLLMClient` (already built in Phase 5) is constructed with the
API key and `settings.llm_model`; `settings.llm_max_tokens` and
`settings.llm_timeout_seconds` are passed into `Agent`'s constructor and
from there into every `LLMRequest`. No credential, model name, or timeout
is ever hard-coded. Constructing `anthropic.Anthropic(api_key=...)` makes
no network call by itself (verified directly) — only `.messages.create()`
does — so `build_agent`'s success path is testable with a syntactically
plausible but fake key and no network access.

**Integration defect found and fixed**: `Settings.llm_max_tokens` and
`Settings.llm_timeout_seconds` existed since Phase 5 but were never
actually threaded into `LLMRequest` construction — `Agent` used
module-level defaults regardless of configuration. This was found during
Phase 6's "inspect before coding" pass, not reported by a test failure.
Fix: `Agent.__init__` gained `max_tokens`/`timeout_seconds` keyword
parameters (defaulting to the prior constants, so all 26 existing Phase 5
orchestration tests still pass unchanged), and `_generate_and_validate`
now builds `LLMRequest` from `self._max_tokens`/`self._timeout_seconds`
instead of the module constants. A dedicated regression test
(`test_agent_threads_configured_max_tokens_and_timeout_into_the_llm_request`
in `tests/unit/test_agent_orchestration.py`) asserts the values actually
reach the request. This is the only change made to a prior phase's code
in Phase 6, and it is a strict widening (new optional constructor
parameters with backward-compatible defaults), not a redesign.

### 19.4 End-to-end request flow

Unchanged from Phase 5's `Agent.handle_message`, now exercised for real
through the CLI: `run_chat_loop` reads one line, strips and checks it for
exit words or emptiness, then calls `_safe_handle_message(agent,
session_id, message)`, which calls `agent.handle_message(session_id,
message)` and converts any *unanticipated* exception into `None` (logged
in full, never shown to the user) so one bad turn cannot crash the
session. Every anticipated failure mode (LLM timeout, provider error,
malformed structured output, retrieval failure, order-lookup failure) is
already converted by `Agent` itself into a safe, deterministic
`AgentResponse` — the CLI's `except Exception` boundary is a documented
last resort for genuinely unexpected failures (e.g. an index file removed
mid-session), not the primary error-handling mechanism.

### 19.5 Output rendering

`render_response(response: AgentResponse) -> str` is presentation-only: it
never re-derives, reformats, or re-inspects citations, and never branches
on the *text* of `response.answer`. `response.answer` already contains an
application-rendered `"Sources:\n- filename — heading"` block whenever
`response.citable_sources` is non-empty (baked in by
`orchestration._append_citations`, Phase 5) — `render_response` only
appends a fixed `[A human team member may follow up on this.]` note when
`response.handoff` is true. The CLI never prints raw LLM output: the only
text ever passed to `print_output` is `response.answer`/`render_response`,
both sourced from the validated `AgentResponse`, never from an LLM
response object directly.

### 19.6 Error and interrupt handling

- `Ctrl+C` and EOF (`Ctrl+D` / piped-input exhaustion) both end the loop
  cleanly via a shared `except (EOFError, KeyboardInterrupt)` clause,
  printing "Goodbye." and returning 0 — not a stack trace.
- Empty input reprompts without calling the agent at all.
- A missing `ANTHROPIC_API_KEY` fails `chat` immediately with a concise,
  actionable message on stderr and exit code 1, before touching the index
  or order dataset.
- A missing/corrupt index or orders file fails `chat` with a distinct,
  actionable message (pointing at `ingest`) and exit code 1 — kept
  separate from the credential-error path (`ConfigurationError` vs.
  `IngestionError`/`OrderError`) so the two failure classes give
  different, correct guidance.
- An LLM timeout or provider error during a turn never crashes the loop
  or reaches the user as raw exception text; the turn resolves to the
  existing Phase 5 `_LLM_UNAVAILABLE_TEXT` fallback with a handoff, and
  the *next* turn is handled normally — a mid-session provider outage
  degrades one answer, not the whole session.

### 19.7 Security behavior (verified through the CLI, not just unit-level)

All six required regression scenarios are exercised end-to-end through
`run_chat_loop` with a `FakeLLMClient` and the real evidence/order layers
in `tests/unit/test_cli_chat.py`: system-prompt extraction, internal
order-note fields (`risk_score`, internal notes), retrieved-content
prompt injection, an order-note-borne prompt injection via `ORD-1005`,
a bulk-customer-data request, and a fabricated address-change claim. The
bulk-data-request test was corrected mid-implementation: "give me all
customer orders" has no order ID and does not match the intent router's
`\border\b`-based pattern (plural "orders" does not match), so it
legitimately reaches the LLM as a knowledge-route question — the real
security property being tested is that no order *lookup* ever occurs for
that route, so there is structurally no customer order data anywhere in
the pipeline for that turn to leak, which the test verifies directly
against both the LLM prompt and the CLI output rather than assuming
outright refusal.

### 19.8 Observability

Structured JSON logging (`logging_setup.setup_logging`) is initialized
once at CLI startup from `settings.log_level`. A dedicated CLI test drives
a full turn against `ORD-1005` (the order with the most sensitive fields
in the dataset) with logging captured, and asserts none of the forbidden
fields (customer email, internal notes, risk score, etc.) appear in any
emitted log record — the same structural allowlist boundary from Phase 4
holds under real end-to-end CLI execution, not just in isolated unit
tests.

### 19.9 Tests

`tests/unit/test_cli_chat.py` (44 tests, network-free, no real Anthropic
API): `build_agent` success/failure paths (missing key, missing index,
missing orders file); `render_response` (plain answer, with citations,
with handoff note); chat-loop mechanics (banner, exit/quit case-
insensitivity, EOF, Ctrl+C, empty input, multi-turn, the last-resort
exception boundary); end-to-end order flow (valid/lowercase/missing/
unknown/malformed IDs, a follow-up turn, cancelled/returned stale-field
suppression, null ETA never fabricated); end-to-end RAG flow (exact,
paraphrased, and product questions, international shipping, a Canada
follow-up, TrailPlus, a superseded-policy question, insufficient
evidence, and the genuine Breeze Tumbler conflict); the five security
regressions above; the observability test; and `main()` dispatch. LLM
failures are scripted via `FakeLLMClient([LLMTimeoutError(...)])` /
`FakeLLMClient([LLMProviderError(...)])`, asserting a safe handoff message
reaches the user with no exception text, and that the session survives to
handle a subsequent turn normally.

`tests/unit/test_agent_orchestration.py` gained one regression test for
the `max_tokens`/`timeout_seconds` wiring fix described in §19.3.

Full suite after Phase 6: 502 tests passed (457 from Phases 2-5, plus 1
orchestration regression test, plus 44 CLI tests). `ruff check` reports no
lint violations; `mypy` reports no issues across 72 source files.

### 19.10 Genuine architectural risks carried forward

- The unsupported-claim validator's curated term list (§18) is unchanged
  and still the main precision gap in response hardening — Phase 6 adds
  no general entailment check.
- `SessionStore` remains in-process, in-memory (Phase 5) — a CLI process
  restart loses all session state, which is acceptable for a single
  long-lived interactive session but would not survive a multi-process
  deployment. Out of scope for Phase 6, unchanged from Phase 5.
- The CLI has no retry/backoff around LLM provider errors — a timeout or
  5xx fails that turn once and hands off, rather than retrying. This
  matches "do not paper over a fabricated action or a genuine provider
  failure" but is worth flagging explicitly as a product-level choice
  rather than an oversight.

## 20. Phase 7 as-built: evaluation framework, adversarial testing, bug diary

Phase 7 adds a repeatable, deterministic evaluation suite over the
complete, real pipeline built in Phases 2-6, and uses it as intended —
to find genuine defects, fix their root causes, and prove the fix with a
regression test. See `docs/evaluation-plan.md` for the full evaluator
design; this section covers what evaluation actually found and changed.

### 20.1 Evaluation command and design summary

`python -m aster_row_agent.cli eval` — one command, no API key or
network access required. Runs the real `Agent` (real embeddings, real
knowledge base, real `data/orders.json`) with a scripted `FakeLLMClient`
against 27 cases (15 supplied visible cases, unmodified, plus 12
original cases in `evaluation/custom-cases.json`), reporting per-case and
per-category PASS/FAIL plus an optional JSON artifact. Full design
rationale — why the core suite stays LLM-free, the assertion mechanism
table, the multi-session case format, and the evaluator's own
data-handling discipline — is in `docs/evaluation-plan.md` §§2-9, not
duplicated here.

### 20.2 Bug diary

Four genuine issues were found by actually running the evaluation suite
against the real pipeline (not invented to satisfy a quota). Three were
root-caused and fixed with a regression test each; the fourth was
root-caused and deliberately left as a documented limitation rather than
patched with a narrow, low-confidence heuristic. At least one (BUG-002,
and independently BUG-003) was found through an original combination/
multi-turn case, beyond the exact wording of any supplied visible case.

---

**BUG-001 — hedged fact-sensitive answer did not force handoff**
Category: groundedness / abstention

*Reproduction*: the visible case `insufficient-information` asks "Are
all fabrics and adhesives in your bags vegan?" This message retrieves
genuinely relevant, `ANSWERABLE` evidence about bag materials in
general — but that evidence never establishes "vegan" specifically. A
correctly-hedged model answer ("I don't have confirmation that... ")
passed response validation cleanly (the answer is honest, not wrong),
but `AgentResponse.handoff` came back `False`, because
`decide_pre_llm_handoff` only forces handoff from the evidence
*bundle's* disposition (`INSUFFICIENT_EVIDENCE`/`AUTHORITATIVE_CONFLICT`/
`NON_AUTHORITATIVE_ONLY`), and this bundle's disposition was
`ANSWERABLE`.

*Expected*: `handoff: true` (the visible case's own requirement, and a
direct reading of the assignment's "recommend human assistance when...
the data is insufficient" rule).

*Actual*: `handoff: false` — verified directly against the real `Agent`,
not inferred.

*Root cause*: bundle-level evidence sufficiency and per-answer fact
sufficiency are different questions that the handoff logic conflated.
`validation.py`'s existing `_unsupported_claim_issue` (the Phase 3/5
"vegan gap" guard) already detects exactly this situation — a
fact-sensitive term the question asks about that the evidence never
establishes — but only used that signal to reject a *confident, wrong*
claim; a correctly-hedged answer produced no signal at all for
orchestration to act on.

*Fix*: `validation.py` gained `_fact_sensitive_hedge_detected` (extracted
from the same `_unestablished_fact_sensitive_terms` helper the existing
issue-check uses, so the two can never disagree) and a
`ValidationResult.fact_sensitive_hedge` field. `orchestration.py`'s
`_generate_and_validate` now computes
`handoff = pre_llm_handoff or validation.fact_sensitive_hedge`. Narrow
and additive: it only fires when a curated fact-sensitive term
(vegan/organic/sustainable/...) is both asked about and correctly
hedged on; every other path is unchanged.

*Regression*: `test_bug001_hedged_fact_sensitive_answer_still_forces_handoff`
in `tests/unit/test_agent_orchestration.py`; also covered end-to-end by
the `insufficient-information` visible case in the eval suite.

*Result*: PASS (eval suite: baseline FAIL → final PASS; full pytest
suite: 505/505 including the new test).

---

**BUG-002 — malformed order ID silently dropped in a combined order+knowledge message**
Category: tool-reliability (found via an original combination case,
beyond visible-case wording)

*Reproduction*: "Can I return ORD-ABCD? Also, what is your return
policy?" — an order-ID-shaped candidate (`ORD-ABCD`) that fails
normalization, combined with a policy question.

*Expected*: the customer should be told their order reference looks
invalid (mirroring the pure-order-question path's
`_MALFORMED_ORDER_ID_TEXT`), in addition to getting the policy answer
they also asked for.

*Actual*: the response answered only the policy half; the LLM was never
even told an invalid order ID had been supplied. Traced to
`orchestration.py`'s `_maybe_lookup_order`: for any route other than a
*pure* `ORDER_LOOKUP`, the function returns `(None, False, False)`
immediately once `resolved_order_id` is `None`, before ever checking
`routing.order_id_candidate` — so the malformed-ID signal that routing.py
*does* capture is discarded for every combined order+knowledge message,
silently.

*Root cause*: the malformed-ID clarification path was implemented only
for the deterministic pure-order-question short-circuit; the combined
route has no equivalent path at all, deterministic or LLM-facing.

*Fix*: `prompts.py`'s `_render_response_requirements` (the same
mechanism already used for "unsupported action" and
"non-authoritative-only" notices) now adds an explicit requirement
whenever `routing.order_id_candidate is not None and
routing.resolved_order_id is None` — this condition can only be reached
for the combined route, since a pure order question with a malformed
candidate already short-circuits earlier. This preserves the knowledge
answer while making the invalid-ID signal visible to the model, instead
of forcing a full short-circuit that would throw away the policy answer
the customer also asked for.

*Regression*: `test_bug002_malformed_order_id_surfaced_in_combined_order_and_knowledge_message`
in `tests/unit/test_agent_orchestration.py`, and the
`malformed-id-combined-with-policy` custom eval case — which asserts
`llm_prompt_must_include: ["ORD-ABCD", "not a valid order ID"]` against
the *actual prompt sent*, not just the (hand-scripted) final answer text.
This distinction mattered in practice: an earlier version of this eval
case checked only `must_include` on the answer and passed even with the
bug present, since the scripted test-double answer already contained the
right words regardless of what the application actually told it — a
reminder that a hand-authored canned response can silently defeat an
assertion aimed at the application layer, not the model.

*Result*: PASS (eval suite: baseline FAIL → final PASS; full pytest
suite: 505/505).

---

**BUG-003 — an unrelated prior topic "rescued" an unrelated, genuinely insufficient question**
Category: multi-turn / groundedness (found via an original multi-turn
case, beyond visible-case wording — and the customer's own reported
"lost context" complaint, in the opposite direction: context bleeding in
where it should not, rather than failing to carry over where it should)

*Reproduction*: Turn 1: "Do you ship internationally?" (answerable,
sets `session.last_topic_hint`). Turn 2, same session: "Which of your
products are vegan?" — standalone, this query is `INSUFFICIENT_EVIDENCE`
(verified directly against the real evidence assembler) and should
short-circuit to the deterministic abstention/handoff response with no
LLM call.

*Expected*: turn 2 behaves identically whether or not turn 1 happened —
`handoff: true`, `used_llm: false`, matching the same disposition the
question gets standalone (the `unsupported-vegan-attribute-standalone`
case).

*Actual*: turn 2 was re-queried with turn 1's unrelated topic hint
prepended ("Do you ship internationally?\nWhich of your products are
vegan?"), which scored as `ANSWERABLE` — the completely unrelated prior
topic silently "rescued" a question that should have triggered
abstention, skipping the required handoff and making an unnecessary
second LLM call.

*Root cause*: `_maybe_assemble_evidence`'s two-pass augmentation retried
*every* insufficient-evidence query with the immediately preceding
message as a hint, with no check that the two messages were actually
related. Notably, no existing test (Phase 5 or 6) ever exercised this
code path succeeding on a genuinely short, ambiguous follow-up — both
existing multi-turn tests' follow-ups ("What about Canada?", "What if
the item is damaged?") turned out to already be `ANSWERABLE` standalone
(verified directly), meaning the mechanism had shipped with zero
demonstrated benefit and, per this finding, a real cost.

*Fix*: augmentation is now also gated on the current message being short
(`len(routing.retrieval_query.split()) <= 5`) — a real ambiguous
follow-up needing prior context is reliably short ("What about Canada?"
is 3 words); a complete, self-contained, unrelated question is not
("Which of your products are vegan?" is 6 words). This is a narrowing,
not a removal: the mechanism still exists for genuinely short fragments,
it simply no longer fires for complete questions that happen to lack
evidence.

*Regression*:
`test_bug003_unrelated_short_topic_does_not_contaminate_a_later_insufficient_question`
in `tests/unit/test_agent_orchestration.py`, and the
`unrelated-topic-then-vegan-question` custom eval case.

*Result*: PASS (eval suite: baseline ERROR — the contamination caused an
*unscripted* second LLM call, which the harness's strict fake-client
correctly surfaced as a hard failure rather than a silent pass — → final
PASS; full pytest suite: 505/505).

---

**BUG-004 — "final-sale-damaged-exception" does not force handoff (documented limitation, not fixed)**
Category: multi-source-grounding / abstention

*Reproduction*: the visible case's own message, "A final-sale bag
arrived with a broken zipper yesterday. Am I completely out of luck?"

*Expected*: `handoff: true` (the visible case's own requirement — doc 04
states "the support agent must not promise that a refund or replacement
has been approved before a human review is completed").

*Actual*: `handoff: false`. Retrieval for this exact query selects doc
04's "Available resolutions" and "Final-sale items" headings, not the
"Reports after seven days" heading that actually carries the
human-review sentence (verified directly: neither this query nor the
`no-lifetime-warranty` query pull in the chunk containing it) — so no
current signal, deterministic or LLM-facing, tells orchestration this
particular answer requires human sign-off.

*Root cause considered and rejected*: a message-level keyword detector
for "arrived damaged"/"broken"/"defective" was considered (matching the
pattern already used for `requests_unsupported_action`), but rejected:
it cannot distinguish an actual, personal damage report ("my bag arrived
broken") from a hypothetical policy question ("what if an item arrives
damaged?") without deeper NLP than a regex can provide, and misfiring
would force an unnecessary handoff on a legitimate, fully-answerable
policy question — the same class of over-fitting the Phase 7 brief
explicitly warns against ("a fix must improve general behavior," not
target one phrase). A document-level marker (flag `04-damaged-or-wrong-items.md`
and `07-warranty.md` as always-handoff) was also rejected: both documents
mix a general-policy section (answerable without handoff — see the
passing `no-lifetime-warranty` case, which cites `07-warranty.md`) with a
claims-process section that does need one, so a whole-document flag would
have broken an already-correct, already-tested case.

*Disposition*: root-caused and explicitly not patched this phase. A
correct general fix needs either finer-grained retrieval (ensuring the
human-review-bearing heading is reliably selected whenever a resolution
is actually being discussed) or a materially better report-vs-hypothetical
intent signal than message keywords — both larger changes than this
phase's "smallest justified fix" scope, and neither would be exercised
by test data broad enough to trust. Documented here and in
`docs/evaluation-plan.md` §11 as a known, understood gap rather than
silently left failing or papered over with a special case.

*Result*: FAIL in both baseline and final runs (unchanged, by design).

### 20.3 Baseline vs. final results

Baseline captured by temporarily reverting the BUG-001/002/003 fixes
(this repository has no prior commits to diff against — see
`docs/evaluation-plan.md` §10 for the exact method) and running the
suite; the fixes were then restored (confirmed byte-identical via the
full pytest suite passing before and after reversion) and the suite
re-run as final.

```
CATEGORY                      BASELINE          FINAL
                             PASS  TOTAL      PASS  TOTAL
------------------------------------------------------------
retrieval                      3     3          3     3
multi-source-grounding         0     1          0     1
conversation                   1     1          1     1
groundedness                   3     3          3     3
tool-use                       4     4          4     4
tool-reliability               4     5          5     5
privacy                        1     1          1     1
prompt-security                2     2          2     2
abstention                     1     2          2     2
source-conflict                2     2          2     2
authority                      1     1          1     1
multi-turn                     1     2          2     2
------------------------------------------------------------
OVERALL                       23    27         26    27
```

The single remaining final-run failure is BUG-004 (§20.2), unchanged
between runs by design. Raw artifacts: `evaluation/results-baseline.json`,
`evaluation/results-final.json`.

### 20.4 Tests added

- `tests/unit/test_agent_orchestration.py`: three regression tests, one
  per fixed bug (§20.2).
- `tests/unit/test_evaluation_runner.py`: two tests exercising the
  evaluator itself — that the full 27-case suite passes except the one
  documented limitation, and that two independent runs produce identical
  pass/fail outcomes (determinism).
- Full suite after Phase 7: **507 tests passed** (502 from Phases 2-6,
  plus 3 bug regression tests, plus 2 evaluation-suite tests). `ruff
  check` reports no lint violations; `mypy` reports no issues across 79
  source files.

### 20.5 Deviations from Phases 1-6

- The only source changes to Phase 1-6 code are the three narrow,
  additive fixes in §20.2 (BUG-001/002/003) — no historical section of
  this document was rewritten, and no earlier phase's tests were
  weakened or deleted (all pre-Phase-7 tests still pass unchanged).
- The Phase 1 evaluation design (`docs/evaluation-plan.md`'s original
  content, now superseded by its Phase 7 as-built rewrite) proposed a
  richer trace/`ResponseEnvelope` object and a `evaluation/runner.py`
  top-level module; the as-built version instead reuses `AgentResponse`
  directly plus two minimal recording subclasses (`harness.py`), and
  lives under `src/aster_row_agent/evaluation/` as a proper package
  wired into the existing `cli.py` entry point — smaller and more
  consistent with the codebase's Phase 2-6 conventions than the original
  Phase 1 sketch, which predated any of that code existing.

### 20.6 Genuine remaining risks (evaluation-specific)

- `must_include_concepts` is a hand-authored keyword approximation, not
  semantic entailment — see `docs/evaluation-plan.md` §11 for the full
  discussion of what this does and does not prove.
- The suite never exercises real model phrasing quality (deliberate, not
  an oversight — `docs/evaluation-plan.md` §3).
- BUG-004 remains a real, understood gap in forcing handoff for
  policy-answerable-but-actually-requires-human-review situations whose
  triggering heading isn't retrieved for the exact query — a genuine
  precision limit of heading-level chunking combined with keyword-free
  intent detection, not a corner case invented for this report.

## 21. Phase 8 addendum: final hardening

Phase 8 is submission preparation (README, demo script, final audits) —
it made no architectural changes. Its mandated end-to-end smoke test
(the 10 scenarios in the Phase 8 brief, run against the real pipeline
with a scripted `FakeLLMClient`, not a live model) did surface one more
genuine bug in the BUG-003 fix, which is recorded here as a direct
continuation of the §20.2 bug diary rather than folded into it, since it
was found in a different phase against different, non-visible-case
wording.

**BUG-005 — a short-but-complete unrelated question still slipped past the BUG-003 word-count guard**
Category: multi-turn

*Reproduction*: from the Phase 8 smoke test's own scenario list — "Can I
put the Breeze Tumbler in the dishwasher?" (sets `session.last_topic_hint`,
disposition `AUTHORITATIVE_CONFLICT`) followed, in the same session, by
"Which products are vegan?" (4 words).

*Expected*: identical behavior to the standalone question — `handoff:
true`, `used_llm: false` (this question alone is `INSUFFICIENT_EVIDENCE`,
verified directly).

*Actual*: the BUG-003 fix's `len(retrieval_query.split()) <= 5` guard let
this phrasing through (4 words), and augmenting it with the unrelated
Breeze Tumbler topic hint scored `AUTHORITATIVE_CONFLICT` — the exact
contamination BUG-003 was meant to close, just with a shorter sentence
than the one BUG-003's fix was validated against ("Which **of your**
products are vegan?", 6 words).

*Root cause*: word count alone cannot distinguish a short genuine
follow-up ("What about Canada?") from a short but complete,
self-contained, unrelated question — both can be short; only the former
actually depends on the prior turn.

*Fix*: `orchestration.py` now also requires the query to contain a
referential/anaphoric marker (`_REFERENTIAL_FOLLOWUP_RE`: "that", "this",
"those", "it", "same", "either", "what about", "how about") before
augmenting — `len(query.split()) <= 5 and _REFERENTIAL_FOLLOWUP_RE.search(query)`.
A genuine follow-up leans on an anaphor because it has no complete
subject of its own; a complete question does not, regardless of length.

*Regression*: `test_bug005_short_unrelated_question_still_does_not_contaminate`
in `tests/unit/test_agent_orchestration.py`, and the
`short-unrelated-question-after-conflict-topic` case added to
`evaluation/custom-cases.json` (13th original case).

*Result*: PASS. Full pytest suite after this fix: **508 passed** (507
from Phases 2-7, plus this one regression test). `ruff check` and `mypy`
remain clean. The evaluation suite grew from 27 to 28 cases; final
result is 27/28 passed (BUG-004 remains the sole, documented failure —
see §20.2). `evaluation/results-baseline.json` (27 cases, captured before
this addition) is preserved unmodified as an honest historical artifact
of the Phase 6→7 transition; it is not retroactively extended to 28.

No other code changes were made in Phase 8. The full final verification
checklist (immutability, secrets, README/documentation accuracy, CI
commands) is reported in the Phase 8 completion report rather than
duplicated here.
