
from types import SimpleNamespace

import pytest

from researchpilot.llm.usage import (
    TokenUsage,
    UsageTracker,
    track_usage,
)


@pytest.fixture
def tracker():
    return UsageTracker()


def test_track_usage_extracts_tokens(tracker):
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
    )

    token_usage = track_usage(response)

    assert token_usage == TokenUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )


def test_track_usage_records_cumulative_totals(tracker):
    tracker.record(
        TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
    )
    tracker.record(
        TokenUsage(
            input_tokens=200,
            output_tokens=80,
            total_tokens=280,
        )
    )

    assert tracker.get_summary() == {
        "total_requests": 2,
        "total_input_tokens": 300,
        "total_output_tokens": 130,
        "total_tokens": 430,
    }


def test_track_usage_returns_none_when_usage_missing():
    response = SimpleNamespace(usage=None)

    assert track_usage(response) is None


def test_track_usage_handles_zero_tokens():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        )
    )

    token_usage = track_usage(response)

    assert token_usage == TokenUsage(0, 0, 0)


def test_tracker_reset(tracker):
    tracker.record(TokenUsage(100, 50, 150))

    tracker.reset()

    assert tracker.get_summary() == {
        "total_requests": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
    }


def test_track_usage_records_in_shared_tracker(monkeypatch):
    import researchpilot.llm.usage as usage_module

    tracker = UsageTracker()
    monkeypatch.setattr(
        usage_module,
        "usage_tracker",
        tracker,
    )

    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
    )

    track_usage(response)

    assert tracker.get_summary() == {
        "total_requests": 1,
        "total_input_tokens": 100,
        "total_output_tokens": 50,
        "total_tokens": 150,
    }
