from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from agentic_trading.market_data import market_data_client as mdc
from agentic_trading.market_data.models import HistoricalBar, Quote

_QUOTE = Quote(
    symbol="AAPL",
    bid_price=1.0,
    ask_price=1.1,
    bid_size=100,
    ask_size=100,
    last_trade_price=1.05,
    updated_at=None,
)
_BAR = HistoricalBar(
    symbol="AAPL",
    begins_at=datetime(2026, 8, 28, 13, 30, tzinfo=UTC),
    open=1.0,
    high=1.1,
    low=0.9,
    close=1.05,
    volume=1000,
)


def test_get_quote_prefers_schwab_when_available(monkeypatch):
    monkeypatch.setattr(mdc.schwab_client, "get_quote", lambda symbol: _QUOTE)

    def _fail_if_called(symbol):
        raise AssertionError("Robinhood fallback should not be called")

    monkeypatch.setattr(mdc.robinhood_client, "get_quote", _fail_if_called)

    assert mdc.get_quote("AAPL") is _QUOTE


def test_get_quote_falls_back_to_robinhood_when_schwab_returns_none(monkeypatch):
    monkeypatch.setattr(mdc.schwab_client, "get_quote", lambda symbol: None)
    monkeypatch.setattr(mdc.robinhood_client, "get_quote", lambda symbol: _QUOTE)

    assert mdc.get_quote("AAPL") is _QUOTE


def test_get_quote_returns_none_when_both_providers_fail(monkeypatch):
    monkeypatch.setattr(mdc.schwab_client, "get_quote", lambda symbol: None)
    monkeypatch.setattr(mdc.robinhood_client, "get_quote", lambda symbol: None)

    assert mdc.get_quote("AAPL") is None


def test_get_5min_historicals_prefers_schwab_when_available(monkeypatch):
    monkeypatch.setattr(
        mdc.schwab_client,
        "get_5min_historicals",
        lambda symbol, start_datetime, end_datetime: [_BAR],
    )

    def _fail_if_called(symbol, span):
        raise AssertionError("Robinhood fallback should not be called")

    monkeypatch.setattr(mdc.robinhood_client, "get_5min_historicals", _fail_if_called)

    bars = mdc.get_5min_historicals("AAPL", span="day")

    assert bars == [_BAR]


def test_get_5min_historicals_falls_back_to_robinhood_when_schwab_is_empty(monkeypatch):
    monkeypatch.setattr(
        mdc.schwab_client,
        "get_5min_historicals",
        lambda symbol, start_datetime, end_datetime: [],
    )
    monkeypatch.setattr(
        mdc.robinhood_client, "get_5min_historicals", lambda symbol, span: [_BAR]
    )

    bars = mdc.get_5min_historicals("AAPL", span="week")

    assert bars == [_BAR]


def test_get_5min_historicals_passes_span_through_to_robinhood_fallback(monkeypatch):
    monkeypatch.setattr(
        mdc.schwab_client,
        "get_5min_historicals",
        lambda symbol, start_datetime, end_datetime: [],
    )
    seen_spans = []
    monkeypatch.setattr(
        mdc.robinhood_client,
        "get_5min_historicals",
        lambda symbol, span: seen_spans.append(span) or [],
    )

    mdc.get_5min_historicals("AAPL", span="week")

    assert seen_spans == ["week"]


def test_get_5min_historicals_rejects_an_unsupported_span():
    with pytest.raises(ValueError):
        mdc.get_5min_historicals("AAPL", span="month")


def test_get_5min_historicals_translates_day_span_to_market_open_start(monkeypatch):
    """span="day" has no Schwab shorthand -- market_data_client must compute an
    explicit start_datetime pinned to today's market open, not e.g. midnight."""
    seen_kwargs = {}

    def _capture(symbol, start_datetime, end_datetime):
        seen_kwargs["start_datetime"] = start_datetime
        seen_kwargs["end_datetime"] = end_datetime
        return [_BAR]

    monkeypatch.setattr(mdc.schwab_client, "get_5min_historicals", _capture)

    mdc.get_5min_historicals("AAPL", span="day")

    settings = mdc.get_settings()
    expected_open = mdc.market_open_today(seen_kwargs["end_datetime"])
    assert seen_kwargs["start_datetime"] == expected_open
    local_open = expected_open.astimezone(ZoneInfo(settings.timezone))
    assert local_open.strftime("%H:%M") == settings.market_open_time


def test_get_5min_historicals_translates_week_span_to_a_multi_day_lookback(monkeypatch):
    seen_kwargs = {}

    def _capture(symbol, start_datetime, end_datetime):
        seen_kwargs["start_datetime"] = start_datetime
        seen_kwargs["end_datetime"] = end_datetime
        return [_BAR]

    monkeypatch.setattr(mdc.schwab_client, "get_5min_historicals", _capture)

    mdc.get_5min_historicals("AAPL", span="week")

    span = seen_kwargs["end_datetime"] - seen_kwargs["start_datetime"]
    assert span.days == mdc._WEEK_SPAN_LOOKBACK_DAYS
