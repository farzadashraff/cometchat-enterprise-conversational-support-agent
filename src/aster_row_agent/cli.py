"""Command-line entry point.

Documented usage:

    python -m aster_row_agent.cli ingest
    python -m aster_row_agent.cli chat

`chat` is a thin interface over the Phase 5 `Agent` — this module contains
no routing, retrieval, sanitization, prompting, or validation logic of its
own. Its only two responsibilities are (1) composing the one, real
`Agent` from `Settings` (`build_agent`) and (2) running an interactive
read-print loop over it (`run_chat_loop`). Both are free functions, not
buried in `main()`, specifically so tests can call `run_chat_loop`
directly with a test-double `Agent` (built with `FakeLLMClient` and the
real evidence/order layers, exactly as `tests/unit/test_agent_orchestration.py`
already does) without needing a real LLM, a real terminal, or `main()`'s
argument parsing at all.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from aster_row_agent.agent.llm import AnthropicLLMClient
from aster_row_agent.agent.models import AgentResponse
from aster_row_agent.agent.orchestration import Agent
from aster_row_agent.agent.session import SessionStore
from aster_row_agent.config import Settings, get_settings
from aster_row_agent.evaluation.reporter import render_console, to_json_dict
from aster_row_agent.evaluation.runner import run_suite
from aster_row_agent.logging_setup import setup_logging
from aster_row_agent.orders.errors import OrderError
from aster_row_agent.orders.repository import OrderRepository
from aster_row_agent.orders.service import OrderLookupService
from aster_row_agent.rag.embeddings import (
    EmbeddingProvider,
    FakeDeterministicEmbeddingProvider,
    FastEmbedProvider,
)
from aster_row_agent.rag.errors import IngestionError
from aster_row_agent.rag.evidence import EvidenceAssembler
from aster_row_agent.rag.ingest import IngestionResult, run_ingestion

logger = logging.getLogger(__name__)

_EXIT_WORDS = {"exit", "quit"}
_HANDOFF_NOTE = "[A human team member may follow up on this.]"


class ConfigurationError(Exception):
    """The application cannot start because required configuration is missing.

    Distinct from `IngestionError`/`OrderError` (a missing/invalid
    *dataset*) — this is specifically for missing *credentials*, so the
    CLI can give the customer-facing-irrelevant, developer-facing-actionable
    guidance appropriate to each ("run ingest first" vs. "set an API key")
    without conflating the two.
    """


# --- ingest command (unchanged from Phase 2) --------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aster-row-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Parse the knowledge base, build chunks + embeddings, write the local index."
    )
    ingest_parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=None,
        help="Override the knowledge-base directory (default: from config/.env).",
    )
    ingest_parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Override the output index directory (default: from config/.env).",
    )
    ingest_parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help="Use the deterministic hash-based embedding provider instead of the "
        "real local model. Fast and fully offline; useful for smoke-testing "
        "the pipeline without downloading model weights.",
    )

    subparsers.add_parser(
        "chat", help="Start an interactive chat session with the Aster & Row support agent."
    )

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run the deterministic evaluation suite (visible + original cases) against "
        "the real Agent pipeline. No API key or network access required.",
    )
    eval_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write machine-readable JSON results to this path in addition to the console report.",
    )
    eval_parser.add_argument(
        "--label",
        type=str,
        default="run",
        help="A short label for this run, stored in the JSON output (e.g. 'baseline', "
        "'final'). Purely descriptive — does not affect grading.",
    )
    return parser


def _resolve_embedding_provider(settings: Settings, *, use_fake: bool) -> EmbeddingProvider:
    if use_fake:
        return FakeDeterministicEmbeddingProvider()
    return FastEmbedProvider(settings.embedding_model_name, settings.embedding_cache_dir)


def _print_report(result: IngestionResult) -> None:
    manifest = result.manifest
    print("Ingestion complete.")
    print(f"  Documents ingested : {manifest.document_count}")
    print(f"  Chunks generated   : {manifest.chunk_count}")
    dims = manifest.embedding_dimensions
    print(f"  Embedding model    : {manifest.embedding_model} ({dims} dims)")
    print(f"  Corpus fingerprint : {manifest.corpus_fingerprint[:16]}...")
    print(f"  Duration           : {manifest.ingestion_duration_ms:.1f} ms")
    print(f"  Index written to   : {result.index_path}")
    print(f"  Manifest written to: {result.index_path.parent / 'manifest.json'}")
    print()
    print("Per-document breakdown:")
    for entry in manifest.documents:
        print(
            f"  - {entry.source_path:45s} status={entry.status:11s} "
            f"audience={entry.audience:9s} authority={entry.policy_authority:9s} "
            f"chunks={entry.chunk_count}"
        )


def _run_ingest_command(args: argparse.Namespace) -> int:
    settings = get_settings()
    overrides: dict[str, Path] = {}
    if args.knowledge_base is not None:
        overrides["knowledge_base_dir"] = args.knowledge_base
    if args.index_dir is not None:
        overrides["index_dir_override"] = args.index_dir
    if overrides:
        settings = settings.model_copy(update=overrides)

    setup_logging(settings.log_level)
    embedding_provider = _resolve_embedding_provider(settings, use_fake=args.fake_embeddings)

    try:
        result = run_ingestion(settings, embedding_provider)
    except IngestionError as exc:
        logger.error("ingestion.failed", extra={"context": {"error": str(exc)}})
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    _print_report(result)
    return 0


# --- chat command (Phase 6) ---------------------------------------------------------------


def build_agent(settings: Settings) -> Agent:
    """The one, real application composition path: Settings -> Agent.

    Composition order matches docs/architecture.md §Phase 6:
    Settings -> (embedding provider -> Retriever/EvidenceAssembler) and
    (OrderRepository -> OrderLookupService) and (AnthropicLLMClient) ->
    Agent -> (caller owns the SessionStore's lifetime).

    The API key is checked first, before any index/dataset loading, so a
    missing-credential failure is immediate and never depends on the
    knowledge base or order dataset happening to be in good shape —
    and so tests can exercise this failure path without needing a real
    index built at all.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "ANTHROPIC_API_KEY is not set. Add it to your environment or a local "
            ".env file (see .env.example) before running 'chat'."
        )

    embedding_provider = FastEmbedProvider(
        settings.embedding_model_name, settings.embedding_cache_dir
    )
    evidence_assembler = EvidenceAssembler.load(settings.index_dir, embedding_provider)
    order_repository = OrderRepository.load(settings.orders_file)
    order_service = OrderLookupService(order_repository)
    llm_client = AnthropicLLMClient(api_key=api_key, model=settings.llm_model)

    return Agent(
        session_store=SessionStore(),
        evidence_assembler=evidence_assembler,
        order_lookup_service=order_service,
        llm_client=llm_client,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def render_response(response: AgentResponse) -> str:
    """Render one `AgentResponse` for display — presentation only.

    `response.answer` already includes an application-rendered "Sources:"
    section when `response.citable_sources` is non-empty (see
    `orchestration._append_citations`) — this function never re-derives
    or reformats citations itself, and never reads `response.answer`'s
    text to decide anything; it only adds a fixed, disposition-driven
    handoff note.
    """
    lines = [response.answer]
    if response.handoff:
        lines.append("")
        lines.append(_HANDOFF_NOTE)
    return "\n".join(lines)


def _safe_handle_message(agent: Agent, session_id: str, message: str) -> AgentResponse | None:
    """The last-resort boundary around one turn.

    `Agent.handle_message` already converts every anticipated failure
    (LLM timeout/provider error/malformed output) into a safe
    `AgentResponse` internally (see docs/architecture.md §Phase 5) — this
    except-Exception is deliberately only for genuinely *unanticipated*
    failures (e.g. an index file removed mid-session), so one bad turn
    can never crash the whole interactive loop. The real exception is
    logged in full via structured logging; nothing from it is ever shown
    to the user.
    """
    try:
        return agent.handle_message(session_id, message)
    except Exception:  # noqa: BLE001 - the documented last-resort CLI boundary
        logger.exception("cli.turn_failed", extra={"context": {"session_id": session_id}})
        return None


def run_chat_loop(
    agent: Agent,
    session_id: str,
    *,
    read_input: Callable[[str], str] = input,
    print_output: Callable[..., None] = print,
) -> int:
    """Run the interactive read-print loop for one session, until exit.

    `read_input`/`print_output` are injected (defaulting to the builtins)
    specifically so tests can drive this loop with scripted input and
    capture output without a real terminal — see
    `tests/unit/test_cli_chat.py`.
    """
    print_output("Aster & Row Support")
    print_output("Type 'exit' or 'quit' to end the session.")
    print_output()

    while True:
        try:
            raw_message = read_input("You: ")
        except (EOFError, KeyboardInterrupt):
            print_output()
            print_output("Goodbye.")
            return 0

        message = raw_message.strip()
        if message.lower() in _EXIT_WORDS:
            print_output("Goodbye.")
            return 0
        if not message:
            print_output("Please type a question, or 'exit' to leave.")
            print_output()
            continue

        response = _safe_handle_message(agent, session_id, message)
        if response is None:
            print_output()
            print_output("Sorry, something went wrong on my end. Please try again.")
            print_output()
            continue

        print_output()
        print_output("Aster & Row:")
        print_output(render_response(response))
        print_output()


def _run_chat_command(args: argparse.Namespace) -> int:
    settings = get_settings()
    setup_logging(settings.log_level)

    try:
        agent = build_agent(settings)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except (IngestionError, OrderError) as exc:
        logger.error("chat.startup_failed", extra={"context": {"error": str(exc)}})
        print(f"Could not start: {exc}", file=sys.stderr)
        print(
            "If the knowledge-base index hasn't been built yet, run "
            "'python -m aster_row_agent.cli ingest' first.",
            file=sys.stderr,
        )
        return 1

    session_id = f"cli-{uuid.uuid4().hex[:12]}"
    return run_chat_loop(agent, session_id)


# --- eval command (Phase 7) --------------------------------------------------------------


def _run_eval_command(args: argparse.Namespace) -> int:
    """Run the deterministic evaluation suite. No `ANTHROPIC_API_KEY` or
    network access is required (see docs/evaluation-plan.md) — every case
    runs against the real evidence/order layers with a scripted
    `FakeLLMClient`.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    case_files = [
        settings.evaluation_dir / "visible-cases.json",
        settings.evaluation_dir / "custom-cases.json",
    ]
    missing = [str(p) for p in case_files if not p.is_file()]
    if missing:
        print(f"Evaluation case file(s) not found: {missing}", file=sys.stderr)
        return 1

    try:
        report = run_suite(case_files, settings, run_label=args.label)
    except IngestionError as exc:
        logger.error("eval.startup_failed", extra={"context": {"error": str(exc)}})
        print(f"Could not run evaluation: {exc}", file=sys.stderr)
        print(
            "If the knowledge-base index hasn't been built yet, run "
            "'python -m aster_row_agent.cli ingest' first.",
            file=sys.stderr,
        )
        return 1

    print(render_console(report))

    if args.out is not None:
        args.out.write_text(json.dumps(to_json_dict(report), indent=2), encoding="utf-8")
        print(f"\nJSON results written to {args.out}")

    return 0 if report.failed_count == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        return _run_ingest_command(args)
    if args.command == "chat":
        return _run_chat_command(args)
    if args.command == "eval":
        return _run_eval_command(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
