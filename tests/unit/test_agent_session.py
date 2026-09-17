from __future__ import annotations

from aster_row_agent.agent.session import SessionStore


def test_get_or_create_creates_a_new_session() -> None:
    store = SessionStore()
    session = store.get_or_create("s1")
    assert session.session_id == "s1"
    assert session.turns == ()
    assert session.last_order_id is None


def test_get_or_create_returns_the_same_session_on_repeat_calls() -> None:
    store = SessionStore()
    first = store.get_or_create("s1")
    store.record_turn("s1", "user", "hello")
    second = store.get_or_create("s1")
    assert second.turns == store.get_or_create("s1").turns
    assert first.session_id == second.session_id


def test_record_turn_appends_in_order() -> None:
    store = SessionStore()
    store.record_turn("s1", "user", "first message")
    session = store.record_turn("s1", "agent", "first reply")
    assert [t.text for t in session.turns] == ["first message", "first reply"]
    assert [t.role for t in session.turns] == ["user", "agent"]


def test_turns_are_bounded_to_max_turns() -> None:
    store = SessionStore(max_turns=4)
    session = store.get_or_create("s1")
    for i in range(10):
        session = store.record_turn("s1", "user", f"message {i}")
    assert len(session.turns) == 4
    assert session.turns[-1].text == "message 9"
    assert session.turns[0].text == "message 6"


def test_turn_text_is_truncated_to_max_turn_chars() -> None:
    store = SessionStore(max_turn_chars=10)
    session = store.record_turn("s1", "user", "this is a very long message that exceeds the limit")
    assert len(session.turns[0].text) == 10


def test_turn_count_is_monotonic_and_independent_of_bounded_turns() -> None:
    store = SessionStore(max_turns=2)
    session = store.get_or_create("s1")
    for i in range(5):
        session = store.record_turn("s1", "user", f"msg {i}")
    assert session.turn_count == 5
    assert len(session.turns) == 2


def test_agent_turns_do_not_increment_turn_count() -> None:
    store = SessionStore()
    store.record_turn("s1", "user", "hi")
    session = store.record_turn("s1", "agent", "hello there")
    assert session.turn_count == 1


def test_set_last_order_id() -> None:
    store = SessionStore()
    session = store.set_last_order_id("s1", "ORD-1007")
    assert session.last_order_id == "ORD-1007"
    session = store.set_last_order_id("s1", None)
    assert session.last_order_id is None


def test_set_last_topic_hint_is_truncated() -> None:
    store = SessionStore(max_topic_hint_chars=5)
    session = store.set_last_topic_hint("s1", "a much longer topic hint than allowed")
    assert session.last_topic_hint == "a muc"


def test_set_last_topic_hint_none_clears_it() -> None:
    store = SessionStore()
    store.set_last_topic_hint("s1", "something")
    session = store.set_last_topic_hint("s1", None)
    assert session.last_topic_hint is None


# --- session isolation (INVARIANT 16) --------------------------------------------------


def test_session_isolation_order_id() -> None:
    store = SessionStore()
    store.set_last_order_id("session-a", "ORD-1001")
    session_b = store.get_or_create("session-b")
    assert session_b.last_order_id is None


def test_session_isolation_turns() -> None:
    store = SessionStore()
    store.record_turn("session-a", "user", "secret question about ORD-1001")
    session_b = store.get_or_create("session-b")
    assert session_b.turns == ()


def test_session_isolation_topic_hint() -> None:
    store = SessionStore()
    store.set_last_topic_hint("session-a", "international shipping")
    session_b = store.get_or_create("session-b")
    assert session_b.last_topic_hint is None


def test_many_sessions_remain_fully_independent() -> None:
    store = SessionStore()
    for i in range(20):
        store.record_turn(f"session-{i}", "user", f"message for session {i}")
        store.set_last_order_id(f"session-{i}", f"ORD-{1000 + i}")
    for i in range(20):
        session = store.get_or_create(f"session-{i}")
        assert session.turns[0].text == f"message for session {i}"
        assert session.last_order_id == f"ORD-{1000 + i}"


# --- no sensitive raw order fields are ever storable in session state ------------------


def test_session_model_has_no_field_capable_of_holding_order_data() -> None:
    """Structural check: Session's fields are limited to what conversational
    continuity actually needs — an order ID string and a topic hint
    string, never an order object of any kind."""
    from aster_row_agent.agent.models import Session

    field_names = set(Session.model_fields.keys())
    assert field_names == {
        "session_id",
        "turns",
        "turn_count",
        "last_order_id",
        "last_topic_hint",
        "created_at",
        "last_active_at",
    }
    # And the order-related field is typed as a bare string, not a model.
    assert Session.model_fields["last_order_id"].annotation == str | None
