# Copyright (c) Microsoft. All rights reserved.

"""Dataset identity survives the terminal rollout projection."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    ("task_input", "expected"),
    [
        ({"data_id": 0}, "0"),
        ({"data_id": 0, "instance_id": "fallback"}, "0"),
        ({"instance_id": 0}, "0"),
        ({"data_id": None, "instance_id": 0}, "0"),
        ({"data_id": "primary", "instance_id": "fallback"}, "primary"),
        ({"data_id": "", "instance_id": "fallback"}, "fallback"),
        ({"data_id": None, "instance_id": "fallback"}, "fallback"),
        ({"data_id": "0"}, "0"),
        ({}, ""),
    ],
)
def test_terminal_projection_preserves_dataset_id(
    client: TestClient, auth_headers: dict[str, str], task_input: dict, expected: str
) -> None:
    created = client.post("/api/rollouts", json=[{"input": task_input}], headers=auth_headers)
    assert created.status_code == 201
    rollout_id = created.json()[0]["rollout_id"]
    for state in ("running", "succeeded"):
        response = client.patch(f"/api/rollouts/{rollout_id}", json={"status": {"state": state}}, headers=auth_headers)
        assert response.status_code == 200

    page = client.get("/api/rollouts/terminal", headers=auth_headers)
    assert page.status_code == 200
    assert page.json() == {
        "items": [{"rollout_id": rollout_id, "state": "succeeded", "data_id": expected, "is_train": True}],
        "next_after": 1,
        "total_terminal": 1,
    }
