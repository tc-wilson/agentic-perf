"""Tests for dispatch failure tracking and crash-loop prevention.

Covers #1108 (unbounded handoff retry loop) and #1109 (benchmark agent
crash-loop without backoff). Verifies that the dispatcher enforces retry
caps and exponential backoff for both handoff recovery and agent dispatch.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from orchestrator.dispatcher import Dispatcher


def _make_dispatcher() -> Dispatcher:
    return Dispatcher("http://store", MagicMock(), MagicMock())


class TestHandoffRetryCap:
    """Handoff recovery retries are bounded (#1108)."""

    def test_retries_are_counted(self) -> None:
        d = _make_dispatcher()
        count = d.record_handoff_retry("PERF-1", "executing_benchmark")
        assert count == 1
        count = d.record_handoff_retry("PERF-1", "executing_benchmark")
        assert count == 2

    def test_exhaustion_after_max_retries(self) -> None:
        d = _make_dispatcher()
        for _ in range(d.MAX_HANDOFF_RETRIES):
            d.record_handoff_retry("PERF-1", "executing_benchmark")
        assert d.handoff_retries_exhausted("PERF-1", "executing_benchmark")

    def test_not_exhausted_before_max(self) -> None:
        d = _make_dispatcher()
        for _ in range(d.MAX_HANDOFF_RETRIES - 1):
            d.record_handoff_retry("PERF-1", "executing_benchmark")
        assert not d.handoff_retries_exhausted("PERF-1", "executing_benchmark")

    def test_clear_resets_count(self) -> None:
        d = _make_dispatcher()
        for _ in range(d.MAX_HANDOFF_RETRIES):
            d.record_handoff_retry("PERF-1", "executing_benchmark")
        d.clear_handoff_retries("PERF-1")
        assert not d.handoff_retries_exhausted("PERF-1", "executing_benchmark")

    def test_different_tickets_independent(self) -> None:
        d = _make_dispatcher()
        for _ in range(d.MAX_HANDOFF_RETRIES):
            d.record_handoff_retry("PERF-1", "executing_benchmark")
        assert not d.handoff_retries_exhausted("PERF-2", "executing_benchmark")

    def test_different_statuses_independent(self) -> None:
        d = _make_dispatcher()
        for _ in range(d.MAX_HANDOFF_RETRIES):
            d.record_handoff_retry("PERF-1", "executing_benchmark")
        assert not d.handoff_retries_exhausted("PERF-1", "awaiting_provision")

    def test_reconcile_prunes_stale_counts(self) -> None:
        d = _make_dispatcher()
        d.record_handoff_retry("PERF-1", "executing_benchmark")
        d.reconcile_handoff_blocked({"PERF-1": "awaiting_customer_guidance"})
        assert not d.handoff_retries_exhausted("PERF-1", "executing_benchmark")

    def test_mark_done_clears_retries(self) -> None:
        d = _make_dispatcher()
        for _ in range(d.MAX_HANDOFF_RETRIES):
            d.record_handoff_retry("PERF-1", "executing_benchmark")
        # mark_done is async — test just the sync clear
        d.clear_handoff_retries("PERF-1")
        assert not d.handoff_retries_exhausted("PERF-1", "executing_benchmark")


class TestDispatchFailureTracking:
    """Dispatch failure backoff and crash-loop cap (#1109)."""

    def test_failure_count_increments(self) -> None:
        d = _make_dispatcher()
        assert d.record_dispatch_failure("PERF-1", "executing_benchmark") == 1
        assert d.record_dispatch_failure("PERF-1", "executing_benchmark") == 2

    def test_exhaustion_after_max_failures(self) -> None:
        d = _make_dispatcher()
        for _ in range(d.MAX_DISPATCH_FAILURES):
            d.record_dispatch_failure("PERF-1", "executing_benchmark")
        assert d.dispatch_failures_exhausted("PERF-1", "executing_benchmark")

    def test_not_exhausted_before_max(self) -> None:
        d = _make_dispatcher()
        for _ in range(d.MAX_DISPATCH_FAILURES - 1):
            d.record_dispatch_failure("PERF-1", "executing_benchmark")
        assert not d.dispatch_failures_exhausted("PERF-1", "executing_benchmark")

    def test_cooldown_active_after_failure(self) -> None:
        d = _make_dispatcher()
        d.record_dispatch_failure("PERF-1", "executing_benchmark")
        assert d.is_dispatch_cooling_down("PERF-1", "executing_benchmark")

    def test_cooldown_not_active_without_failure(self) -> None:
        d = _make_dispatcher()
        assert not d.is_dispatch_cooling_down("PERF-1", "executing_benchmark")

    def test_cooldown_exponential_backoff(self) -> None:
        d = _make_dispatcher()
        d.record_dispatch_failure("PERF-1", "executing_benchmark")
        deadline1 = d._dispatch_cooldowns[("PERF-1", "executing_benchmark")]

        d.record_dispatch_failure("PERF-1", "executing_benchmark")
        deadline2 = d._dispatch_cooldowns[("PERF-1", "executing_benchmark")]

        now = time.monotonic()
        delay1 = deadline1 - now
        delay2 = deadline2 - now
        assert delay2 > delay1

    def test_cooldown_capped(self) -> None:
        d = _make_dispatcher()
        for _ in range(20):
            d.record_dispatch_failure("PERF-1", "executing_benchmark")
        deadline = d._dispatch_cooldowns[("PERF-1", "executing_benchmark")]
        delay = deadline - time.monotonic()
        assert delay <= d.DISPATCH_BACKOFF_CAP + 1.0

    def test_clear_resets_everything(self) -> None:
        d = _make_dispatcher()
        for _ in range(d.MAX_DISPATCH_FAILURES):
            d.record_dispatch_failure("PERF-1", "executing_benchmark")
        d.clear_dispatch_failures("PERF-1")
        assert not d.dispatch_failures_exhausted("PERF-1", "executing_benchmark")
        assert not d.is_dispatch_cooling_down("PERF-1", "executing_benchmark")

    def test_different_tickets_independent(self) -> None:
        d = _make_dispatcher()
        for _ in range(d.MAX_DISPATCH_FAILURES):
            d.record_dispatch_failure("PERF-1", "executing_benchmark")
        assert not d.dispatch_failures_exhausted("PERF-2", "executing_benchmark")

    def test_reconcile_prunes_stale_failures(self) -> None:
        d = _make_dispatcher()
        d.record_dispatch_failure("PERF-1", "executing_benchmark")
        d.reconcile_handoff_blocked({"PERF-1": "awaiting_customer_guidance"})
        assert not d.is_dispatch_cooling_down("PERF-1", "executing_benchmark")
        assert not d.dispatch_failures_exhausted("PERF-1", "executing_benchmark")


class TestBlockHandoffUnfencedHeaders:
    """_block_handoff_failed uses unfenced headers (#1110b)."""

    @pytest.fixture()
    def mock_client(self) -> MagicMock:
        from unittest.mock import AsyncMock

        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        return client

    async def test_no_fencing_headers_in_recovery(
        self,
        mock_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Recovery transitions must not include fencing headers."""
        import orchestrator.main as mod

        monkeypatch.setenv("AGENTIC_PERF_API_TOKEN", "test-token")
        monkeypatch.setenv("AGENTIC_PERF_ORCHESTRATOR_SESSION_ID", "sess-1")
        monkeypatch.setenv("AGENTIC_PERF_ORCHESTRATOR_EPOCH", "42")

        captured_headers: list[dict] = []

        def capturing_client(**kwargs):
            captured_headers.append(dict(kwargs.get("headers", {})))
            return mock_client

        def make_response(status_code=200, json_data=None):
            resp = MagicMock()
            resp.status_code = status_code
            resp.json.return_value = json_data or {}
            return resp

        async def fake_post(url, json=None, **kw):
            return make_response(200)

        async def fake_get(url, **kw):
            return make_response(200, {"status": "executing_benchmark"})

        mock_client.post = fake_post
        mock_client.get = fake_get

        with monkeypatch.context() as m:
            m.setattr(mod, "AuditedAsyncHTTPClient", capturing_client)
            await mod._block_handoff_failed(
                store_url="http://store:8090",
                ticket_id="PERF-FENCE",
                reason="test",
                current_status="executing_benchmark",
            )

        assert len(captured_headers) == 1
        headers = captured_headers[0]
        assert "Authorization" in headers
        assert "X-Agentic-Perf-Orchestrator-Session" not in headers
        assert "X-Agentic-Perf-Orchestrator-Epoch" not in headers


class TestAdvancePlanProvisioningCheck:
    """_advance_plan blocks when provisioning_complete is not set (#1110a)."""

    @pytest.fixture()
    def mock_client(self) -> MagicMock:
        from unittest.mock import AsyncMock

        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        return client

    async def test_blocks_advance_when_provisioning_incomplete(
        self,
        mock_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Plan advancement blocks if provisioning_complete is False."""
        import orchestrator.main as mod

        monkeypatch.setenv("AGENTIC_PERF_API_TOKEN", "test-token")

        ticket_data = {
            "status": "awaiting_provision",
            "custom_fields": {
                "provisioning_complete": False,
                "hosts_provisioned": ["host-1", "host-2"],
                "execution_plan": {
                    "steps": [
                        {
                            "agent_type": "provision",
                            "status": "in_progress",
                        },
                        {
                            "agent_type": "benchmark",
                            "status": "pending",
                        },
                    ],
                    "current_step": 0,
                },
            },
        }

        post_calls: list[tuple[str, dict]] = []

        def make_response(status_code=200, json_data=None):
            resp = MagicMock()
            resp.status_code = status_code
            resp.json.return_value = json_data or ticket_data
            resp.raise_for_status = MagicMock()
            return resp

        async def fake_get(url, **kw):
            return make_response(200, ticket_data)

        async def fake_post(url, json=None, **kw):
            post_calls.append((url, json or {}))
            return make_response(200)

        async def fake_patch(url, json=None, **kw):
            resp = make_response(200)
            return resp

        mock_client.get = fake_get
        mock_client.post = fake_post
        mock_client.patch = fake_patch

        with monkeypatch.context() as m:
            m.setattr(mod, "AuditedAsyncHTTPClient", lambda **kw: mock_client)
            await mod._advance_plan(
                store_url="http://store:8090",
                ticket_id="PERF-PROV",
                completed_status="awaiting_provision",
            )

        transition_posts = [(u, j) for u, j in post_calls if "transition" in u]
        assert len(transition_posts) == 1
        assert transition_posts[0][1]["status"] == "awaiting_customer_guidance"
        assert "provisioning" in transition_posts[0][1]["comment"].lower()

    async def test_allows_advance_when_provisioning_complete(
        self,
        mock_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Plan advancement proceeds when provisioning_complete is True."""
        import orchestrator.main as mod

        monkeypatch.setenv("AGENTIC_PERF_API_TOKEN", "test-token")

        ticket_data = {
            "status": "awaiting_provision",
            "custom_fields": {
                "provisioning_complete": True,
                "hosts_provisioned": ["host-1"],
                "execution_plan": {
                    "steps": [
                        {
                            "id": 0,
                            "agent_type": "provision",
                            "status": "in_progress",
                            "results": {},
                        },
                        {
                            "id": 1,
                            "agent_type": "benchmark",
                            "status": "pending",
                            "params": {},
                            "results": {},
                        },
                    ],
                    "current_step": 0,
                },
            },
        }

        post_calls: list[tuple[str, dict]] = []

        def make_response(status_code=200, json_data=None):
            resp = MagicMock()
            resp.status_code = status_code
            resp.json.return_value = json_data or ticket_data
            resp.raise_for_status = MagicMock()
            return resp

        async def fake_get(url, **kw):
            return make_response(200, ticket_data)

        async def fake_post(url, json=None, **kw):
            post_calls.append((url, json or {}))
            return make_response(200)

        async def fake_patch(url, json=None, **kw):
            return make_response(200)

        mock_client.get = fake_get
        mock_client.post = fake_post
        mock_client.patch = fake_patch

        with monkeypatch.context() as m:
            m.setattr(mod, "AuditedAsyncHTTPClient", lambda **kw: mock_client)
            await mod._advance_plan(
                store_url="http://store:8090",
                ticket_id="PERF-PROV2",
                completed_status="awaiting_provision",
            )

        transition_posts = [(u, j) for u, j in post_calls if "transition" in u]
        assert len(transition_posts) == 1
        assert transition_posts[0][1]["status"] == "executing_benchmark"

    async def test_blocks_when_provisioning_complete_is_truthy_non_bool(
        self,
        mock_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Truthy non-boolean values like 'false' must not pass the gate."""
        import orchestrator.main as mod

        monkeypatch.setenv("AGENTIC_PERF_API_TOKEN", "test-token")

        ticket_data = {
            "status": "awaiting_provision",
            "custom_fields": {
                "provisioning_complete": "false",
                "hosts_provisioned": ["host-1"],
                "execution_plan": {
                    "steps": [
                        {
                            "agent_type": "provision",
                            "status": "in_progress",
                        },
                        {
                            "agent_type": "benchmark",
                            "status": "pending",
                        },
                    ],
                    "current_step": 0,
                },
            },
        }

        post_calls: list[tuple[str, dict]] = []

        def make_response(status_code=200, json_data=None):
            resp = MagicMock()
            resp.status_code = status_code
            resp.json.return_value = json_data or ticket_data
            resp.raise_for_status = MagicMock()
            return resp

        async def fake_get(url, **kw):
            return make_response(200, ticket_data)

        async def fake_post(url, json=None, **kw):
            post_calls.append((url, json or {}))
            return make_response(200)

        async def fake_patch(url, json=None, **kw):
            return make_response(200)

        mock_client.get = fake_get
        mock_client.post = fake_post
        mock_client.patch = fake_patch

        with monkeypatch.context() as m:
            m.setattr(mod, "AuditedAsyncHTTPClient", lambda **kw: mock_client)
            await mod._advance_plan(
                store_url="http://store:8090",
                ticket_id="PERF-PROV-TRUTHY",
                completed_status="awaiting_provision",
            )

        transition_posts = [(u, j) for u, j in post_calls if "transition" in u]
        assert len(transition_posts) == 1
        assert transition_posts[0][1]["status"] == "awaiting_customer_guidance"

    async def test_non_benchmark_successor_not_blocked(
        self,
        mock_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Provision→review should not be blocked by provisioning_complete."""
        import orchestrator.main as mod

        monkeypatch.setenv("AGENTIC_PERF_API_TOKEN", "test-token")

        ticket_data = {
            "status": "awaiting_provision",
            "custom_fields": {
                "provisioning_complete": False,
                "hosts_provisioned": [],
                "execution_plan": {
                    "steps": [
                        {
                            "id": 0,
                            "agent_type": "provision",
                            "status": "in_progress",
                            "results": {},
                        },
                        {
                            "id": 1,
                            "agent_type": "review",
                            "status": "pending",
                            "params": {},
                            "results": {},
                        },
                    ],
                    "current_step": 0,
                },
            },
        }

        post_calls: list[tuple[str, dict]] = []

        def make_response(status_code=200, json_data=None):
            resp = MagicMock()
            resp.status_code = status_code
            resp.json.return_value = json_data or ticket_data
            resp.raise_for_status = MagicMock()
            return resp

        async def fake_get(url, **kw):
            return make_response(200, ticket_data)

        async def fake_post(url, json=None, **kw):
            post_calls.append((url, json or {}))
            return make_response(200)

        async def fake_patch(url, json=None, **kw):
            return make_response(200)

        mock_client.get = fake_get
        mock_client.post = fake_post
        mock_client.patch = fake_patch

        with monkeypatch.context() as m:
            m.setattr(mod, "AuditedAsyncHTTPClient", lambda **kw: mock_client)
            await mod._advance_plan(
                store_url="http://store:8090",
                ticket_id="PERF-PROV-REVIEW",
                completed_status="awaiting_provision",
            )

        transition_posts = [(u, j) for u, j in post_calls if "transition" in u]
        assert len(transition_posts) == 1
        assert transition_posts[0][1]["status"] == "awaiting_review"
