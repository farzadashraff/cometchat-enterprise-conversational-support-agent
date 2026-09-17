"""Bounded, in-memory session store.

Explicitly NOT long-term memory (per the Phase 5 brief): each session is
capped at `max_turns` turns, each turn's text capped at `max_turn_chars`,
and a session only ever stores an order *ID string* and a short topic
*hint string* — never order data, never retrieved document text, never
customer PII (INVARIANT 17).

Sessions are strictly isolated by `session_id`: this store is a plain
dict keyed by session_id with no shared mutable state between entries,
so data from one session structurally cannot appear in another
(INVARIANT 16) — verified directly in
`tests/unit/test_agent_session.py::test_session_isolation`.

A database-backed or persistent store was deliberately not built (the
assignment does not ask for durability across process restarts, and
adding one would be exactly the kind of unnecessary infrastructure
Phase 5's scope guard rules out).
"""

from __future__ import annotations

from datetime import UTC, datetime

from aster_row_agent.agent.models import Session, Turn, TurnRole

DEFAULT_MAX_TURNS = 8
DEFAULT_MAX_TURN_CHARS = 2000
DEFAULT_MAX_TOPIC_HINT_CHARS = 300


class SessionStore:
    """An in-memory, bounded, isolated session store."""

    def __init__(
        self,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_turn_chars: int = DEFAULT_MAX_TURN_CHARS,
        max_topic_hint_chars: int = DEFAULT_MAX_TOPIC_HINT_CHARS,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self._max_turns = max_turns
        self._max_turn_chars = max_turn_chars
        self._max_topic_hint_chars = max_topic_hint_chars

    def get_or_create(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is not None:
            return session
        now = datetime.now(UTC)
        session = Session(session_id=session_id, created_at=now, last_active_at=now)
        self._sessions[session_id] = session
        return session

    def record_turn(self, session_id: str, role: TurnRole, text: str) -> Session:
        """Append one turn, bounded by `max_turns` (oldest dropped first)
        and `max_turn_chars` (text truncated, never the whole conversation
        stored unbounded)."""
        session = self.get_or_create(session_id)
        truncated_text = text[: self._max_turn_chars]
        new_turn = Turn(role=role, text=truncated_text, timestamp=datetime.now(UTC))
        turns = (*session.turns, new_turn)[-self._max_turns :]
        turn_count = session.turn_count + 1 if role == "user" else session.turn_count
        return self._update(session, turns=turns, turn_count=turn_count)

    def set_last_order_id(self, session_id: str, order_id: str | None) -> Session:
        session = self.get_or_create(session_id)
        return self._update(session, last_order_id=order_id)

    def set_last_topic_hint(self, session_id: str, hint: str | None) -> Session:
        session = self.get_or_create(session_id)
        truncated = hint[: self._max_topic_hint_chars] if hint is not None else None
        return self._update(session, last_topic_hint=truncated)

    def _update(self, session: Session, **changes: object) -> Session:
        updated = session.model_copy(update={**changes, "last_active_at": datetime.now(UTC)})
        self._sessions[session.session_id] = updated
        return updated
