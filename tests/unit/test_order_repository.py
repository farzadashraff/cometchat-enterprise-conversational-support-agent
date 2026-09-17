from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aster_row_agent.orders.errors import (
    DuplicateOrderIdError,
    OrderDatasetNotFoundError,
    OrderDatasetParseError,
    OrderDatasetValidationError,
)
from aster_row_agent.orders.repository import OrderRepository

EXPECTED_ORDER_COUNT = 12

_VALID_ORDER: dict[str, Any] = {
    "order_id": "ORD-1000",
    "customer": {
        "name": "Test Customer",
        "email": "test@example.test",
        "shipping_address": "123 Fake Street",
    },
    "membership_tier": "standard",
    "items": [{"sku": "SKU-1", "name": "Test Item", "quantity": 1, "final_sale": False}],
    "placed_at": "2026-08-11T15:05:00Z",
    "status": "shipped",
    "status_updated_at": "2026-08-14T20:40:00Z",
    "shipped_at": None,
    "delivered_at": None,
    "carrier": None,
    "tracking_number": None,
    "estimated_delivery": None,
    "customer_safe_message": "In transit.",
    "internal": {"risk_score": 1, "warehouse_note": "x", "support_tags": []},
}


def _write_dataset(
    path: Path, orders: list[dict[str, Any]], snapshot_at: str = "2026-08-15T12:00:00Z"
) -> None:
    payload = {"dataset_name": "Test", "snapshot_at": snapshot_at, "orders": orders}
    path.write_text(json.dumps(payload))


# --- real dataset integration --------------------------------------------------------


def test_load_real_dataset_loads_all_orders(real_orders_file: Path) -> None:
    repo = OrderRepository.load(real_orders_file)
    assert len(repo) == EXPECTED_ORDER_COUNT


def test_load_real_dataset_captures_snapshot_at(real_orders_file: Path) -> None:
    repo = OrderRepository.load(real_orders_file)
    assert repo.snapshot_at.isoformat() == "2026-08-15T12:00:00+00:00"


def test_get_by_order_id_returns_none_for_unknown_id(
    real_order_repository: OrderRepository,
) -> None:
    assert real_order_repository.get_by_order_id("ORD-9999") is None


def test_get_by_order_id_returns_raw_record_for_known_id(
    real_order_repository: OrderRepository,
) -> None:
    record = real_order_repository.get_by_order_id("ORD-1007")
    assert record is not None
    assert record.order_id == "ORD-1007"
    assert record.status == "shipped"


def test_repository_load_does_not_modify_the_source_file(real_orders_file: Path) -> None:
    before = real_orders_file.read_bytes()
    OrderRepository.load(real_orders_file)
    after = real_orders_file.read_bytes()
    assert before == after


# --- error handling ------------------------------------------------------------------


def test_load_missing_file_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(OrderDatasetNotFoundError):
        OrderRepository.load(tmp_path / "does-not-exist.json")


def test_load_invalid_json_raises_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    path.write_text("{not valid json")
    with pytest.raises(OrderDatasetParseError):
        OrderRepository.load(path)


def test_load_non_object_top_level_raises_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(OrderDatasetParseError):
        OrderRepository.load(path)


def test_load_missing_snapshot_at_raises_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    path.write_text(json.dumps({"dataset_name": "Test", "orders": []}))
    with pytest.raises(OrderDatasetParseError):
        OrderRepository.load(path)


def test_load_missing_orders_array_raises_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    path.write_text(json.dumps({"dataset_name": "Test", "snapshot_at": "2026-08-15T12:00:00Z"}))
    with pytest.raises(OrderDatasetParseError):
        OrderRepository.load(path)


def test_load_orders_not_a_list_raises_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    payload = {
        "dataset_name": "Test",
        "snapshot_at": "2026-08-15T12:00:00Z",
        "orders": "not-a-list",
    }
    path.write_text(json.dumps(payload))
    with pytest.raises(OrderDatasetParseError):
        OrderRepository.load(path)


def test_load_malformed_record_raises_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    malformed = dict(_VALID_ORDER)
    del malformed["status"]  # required field missing
    _write_dataset(path, [malformed])
    with pytest.raises(OrderDatasetValidationError) as exc_info:
        OrderRepository.load(path)
    assert len(exc_info.value.errors) == 1


def test_load_reports_every_malformed_record_not_just_the_first(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    bad_one = dict(_VALID_ORDER, order_id="ORD-1001")
    del bad_one["status"]
    bad_two = dict(_VALID_ORDER, order_id="ORD-1002")
    del bad_two["customer"]
    _write_dataset(path, [bad_one, bad_two])
    with pytest.raises(OrderDatasetValidationError) as exc_info:
        OrderRepository.load(path)
    assert len(exc_info.value.errors) == 2


def test_load_invalid_status_type_raises_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    bad = dict(_VALID_ORDER, status=12345)  # status must be a string
    _write_dataset(path, [bad])
    with pytest.raises(OrderDatasetValidationError):
        OrderRepository.load(path)


def test_load_duplicate_order_ids_raises_duplicate_error(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    first = dict(_VALID_ORDER, order_id="ORD-1000")
    second = dict(_VALID_ORDER, order_id="ORD-1000")
    _write_dataset(path, [first, second])
    with pytest.raises(DuplicateOrderIdError) as exc_info:
        OrderRepository.load(path)
    assert "ORD-1000" in exc_info.value.duplicate_ids


def test_duplicate_order_ids_does_not_silently_pick_one(tmp_path: Path) -> None:
    """A DuplicateOrderIdError must prevent the repository from being
    constructed at all — there is no partial/best-effort repository."""
    path = tmp_path / "orders.json"
    first = dict(_VALID_ORDER, order_id="ORD-1000", status="shipped")
    second = dict(_VALID_ORDER, order_id="ORD-1000", status="cancelled")
    _write_dataset(path, [first, second])
    with pytest.raises(DuplicateOrderIdError):
        OrderRepository.load(path)


def test_valid_minimal_dataset_loads_successfully(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    _write_dataset(path, [dict(_VALID_ORDER)])
    repo = OrderRepository.load(path)
    assert len(repo) == 1
    assert repo.get_by_order_id("ORD-1000") is not None


def test_empty_orders_array_loads_successfully_as_empty_repository(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    _write_dataset(path, [])
    repo = OrderRepository.load(path)
    assert len(repo) == 0
    assert repo.get_by_order_id("ORD-1000") is None
