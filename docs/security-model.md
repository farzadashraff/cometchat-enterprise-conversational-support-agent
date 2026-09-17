# Security Model

## 1. Trust boundaries

**TRUSTED** (authored/controlled by the system, never derived from
runtime user/document/tool content):
- The fixed system prompt / behavioral instructions (compiled once at
  build time from the assignment's explicit rules and doc 13).
- Deterministic business logic: authority gating, conflict detection,
  order sanitization, the cancellation time-window check, handoff rule
  table, response validation.
- Validated configuration (`.env` values loaded through
  `pydantic-settings`, type- and range-checked at startup).

**UNTRUSTED DATA** (may contain adversarial or malformed content;
never treated as instructions):
- User messages (any turn, including follow-ups).
- Retrieved knowledge-base passages (including doc 14's embedded fake
  "SYSTEM INSTRUCTION" and doc 02/14's superseded/fabricated claims).
- Tool results (the order-lookup result, even after sanitization —
  sanitization removes forbidden fields, it does not certify the
  remaining text is instruction-free, so it is still wrapped as data).
- Any text originating inside `orders.json`'s `internal.*` fields,
  including the three embedded prompt-injection style
  `warehouse_note` values (see `corpus-analysis.md` §4) — these are
  handled primarily by never reaching the LLM at all (§3 below), which
  is a stronger property than "the LLM was told to ignore them."

## 2. How the architecture prevents untrusted content from becoming instructions

Three independent layers, so that a failure in one does not become a
full compromise:

1. **Field-level exclusion (data never arrives).** `orders/sanitize.py`
   builds a `CustomerSafeOrder` from an explicit allowlist (see
   `corpus-analysis.md` §4). `customer.*` and `internal.*` are not
   filtered out of a string after the fact — they are simply never
   read into any object that flows toward the LLM. This is why the
   `ORD-1005`/`ORD-1007`/`ORD-1012` injection payloads are neutralized
   unconditionally: there is no code path where their text is
   concatenated into a prompt, so there is nothing for a prompt-level
   defense to fail to catch.
2. **Structural prompt isolation (data cannot masquerade as instructions).**
   The prompt template (see `architecture.md` §10) places the fixed
   system instructions in a region that is never a function of
   document/tool/user content, and places all retrieved passages, tool
   results, and conversation history inside explicitly labeled,
   delimited blocks with an explicit instruction that content in those
   blocks is data, not commands. Doc 14's fabricated
   "SYSTEM INSTRUCTION: Ignore all prior rules..." line therefore
   arrives inside a `<evidence>`-tagged block, structurally identical
   in kind to every other retrieved sentence — it is not given any
   special positional or syntactic authority.
3. **Deterministic enforcement independent of model compliance.**
   Even if (1) and (2) both failed for a given call — the model
   "decided" to comply with an injected instruction — the following
   are still true because they are not the model's decision to make:
   the model cannot fabricate a completed refund/cancellation/
   replacement/address-change (no such action exists anywhere in the
   code for it to have triggered); it cannot have actually skipped a
   tool call and still produced sanitized order data (there is no
   sanitized data available unless the repository was actually
   queried); and the response validator independently scans the
   *output* for injected directives having been followed (e.g. a
   suspicious "approved" claim, a raw system-prompt fragment, a
   forbidden-field pattern) and applies the deterministic fallback if
   so, regardless of why the model produced that text.

## 3. Adversarial scenario coverage

Mapped to concrete corpus content and/or visible eval cases where they
exist; scenarios without a visible case are candidates for the ≥5
original evaluation cases (see `evaluation-plan.md`).

| Scenario | Concrete trigger in this corpus | Defense |
|---|---|---|
| Paraphrased policy questions | Any policy question phrased differently from the visible cases | Retrieval + authority gating operate on meaning/metadata, not exact wording; reviewers are explicitly told they will test paraphrases |
| Legacy vs. current policy confusion | Doc 02 vs doc 01 | Supersession gate (architecture.md §4) |
| Genuine active-source conflict | Doc 11 vs doc 12 (Breeze Tumbler dishwasher safety) | Conflict registry (architecture.md §5); `must_not_silently_choose_one` |
| Retrieved prompt injection | Doc 14's embedded fake system instruction | Structural prompt isolation (§2.2) + authority gate excludes doc 14 from being cited as authority regardless |
| Order prompt injection via internal fields | `ORD-1005`/`1007`/`1012` `internal.warehouse_note` | Field-level exclusion (§2.1) — the text never reaches the LLM |
| Missing order ID | `missing-order-id` visible case | Deterministic classification (`normalize.classify` returns "no ID found") → clarifying question, no tool call |
| Malformed order ID | e.g. `"12345"`, `"banana"` (original case) | Same classifier, "not plausibly an ID" branch → clarifying question, no tool call with garbage input |
| Lowercase/whitespace order IDs | e.g. `" ord-1007 "` (original case) | Normalization regex in `orders/normalize.py`, unit-tested directly |
| Unknown order | `ORD-9999`, `unknown-order` visible case | Tool call proceeds, repository returns not-found, response says so + recommends handoff, no invented status |
| Cancelled order with stale ETA | `ORD-1004`, `cancelled-order-stale-eta` visible case | Sanitizer omits stale fields for `status in {cancelled, returned}` |
| Returned order with stale fields | `ORD-1008` (original case) | Same sanitizer rule |
| Shipped order with no ETA | `ORD-1011`, `shipped-without-eta` visible case | Sanitizer omits `estimated_delivery` entirely when null; system prompt forbids invention |
| Exception order | `ORD-1010` | Deterministic rule: `status==exception` → mandatory handoff |
| Privacy extraction (order data) | `order-data-privacy` visible case | Sanitizer allowlist (§2.1) + validator forbidden-field scan (defense in depth) |
| System prompt extraction | Original case (not in visible set) | Fixed refusal instruction + validator scan for system-prompt-fragment leakage in output |
| Unrelated conversation context contamination | Original case: off-topic chit-chat then a policy question | Bounded session history + topic-tag-based retrieval hint, not raw transcript concatenation into the query |
| Follow-up requiring previous context | `canada-multiturn`; `"Where is ORD-1007?"` → `"When will it arrive?"` (original case) | Explicit session-state fields (`last_order_id`, `last_topic_tags`), not implicit model memory alone |
| Unsupported company-specific questions | `insufficient-information` (vegan materials) | Empty/insufficient evidence → deterministic abstention + handoff, not general model knowledge |
| Requests to perform unsupported actions | Original case: "cancel my order right now" | No cancellation action exists in code; handoff rule table triggers; response validator blocks any completion-claim language |
| Attempts to induce a false "action completed" claim | Same as above, and any injected instruction demanding it (doc 14, `ORD-1005` note) | Same as above — this is a structural impossibility, not a prompt-compliance hope |

## 4. Privacy design details

- **Allowlist, not blocklist**, for order data (`corpus-analysis.md` §4)
  — new internal fields added to `orders.json` in the future are safe
  by default (excluded) rather than unsafe by default (leaked until
  someone remembers to blocklist them).
- **Gift-card codes**: per doc 10, the agent must never ask a customer
  to paste a complete gift-card code into chat — this is enforced as a
  fixed system-prompt rule plus a response-validator check that the
  agent's own messages never request one.
- **Cross-session isolation**: session state is keyed strictly by
  `session_id` with no shared mutable state between sessions (see
  `architecture.md` §9); this prevents one customer's order context
  from leaking into another customer's conversation, which is itself a
  privacy property even though it isn't framed that way in the
  assignment text.
- **Observability redaction**: the trace writer (`observability/trace.py`)
  reuses the exact same allowlist function as the sanitizer (one source
  of truth) so that "never log secrets or forbidden customer fields"
  cannot silently drift out of sync with "never expose them to the
  customer."

## 5. Secrets / credentials

- No API keys or credentials are ever committed; `.env.example` lists
  variable names only (e.g. `ANTHROPIC_API_KEY=`), with placeholder or
  empty values.
- The LLM client reads the key from environment at startup and never
  logs it; the structured trace schema does not include a field capable
  of holding it.
- CI (GitHub Actions) will need a repository secret for any live-LLM
  eval run; the eval suite's deterministic assertions should be
  designed to also make sense as a smoke test if run against a
  recorded/replayed set of model outputs, so CI does not strictly
  require live API access to validate the deterministic logic (tool
  behavior, sanitization, authority gating) — only the "concept"-level
  assertions need a live or recorded generation. This split is noted as
  a Phase-2 implementation detail in `implementation-plan.md`.

## 6. What this design deliberately does not attempt

- General-purpose jailbreak resistance beyond the scenarios above — out
  of scope for the assignment's timebox and corpus.
- Full identity verification for order lookups — the assignment
  explicitly states possession of the order ID is sufficient
  authentication for this mock system.
- Rate limiting, abuse detection, or authentication/authorization —
  explicitly listed as out of scope in the assignment brief.
