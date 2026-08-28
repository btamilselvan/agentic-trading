from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from agentic_trading.market_data import schwab_client as sc


@dataclass
class _FakeSettings:
    schwab_client_id: str | None = "client-id"
    schwab_client_secret: str | None = "client-secret"
    schwab_token_path: str = ".secrets/schwab_token.json"


class _FakeResponse:
    def __init__(self, status_code: int, payload: object = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        return self._payload


class _FakeSchwabClient:
    """Plain object standing in for a schwab-py client -- unlike `object()`, this
    supports monkeypatch.setattr assigning per-test methods onto an instance."""


@pytest.fixture(autouse=True)
def _reset_client_state():
    """`_get_client()` caches the schwab-py client (and any init failure's retry
    cooldown) in module-level globals -- isolate tests from each other."""
    sc._client = None
    sc._client_init_retry_after = None
    yield
    sc._client = None
    sc._client_init_retry_after = None


def test_get_client_returns_none_when_credentials_not_configured(monkeypatch):
    monkeypatch.setattr(
        sc,
        "get_settings",
        lambda: _FakeSettings(schwab_client_id=None, schwab_client_secret=None),
    )

    assert sc._get_client() is None


def test_get_client_caches_the_constructed_client(monkeypatch):
    monkeypatch.setattr(sc, "get_settings", lambda: _FakeSettings())
    calls = []

    def fake_client_from_token_file(*, token_path, api_key, app_secret):
        calls.append((token_path, api_key, app_secret))
        return object()

    monkeypatch.setattr(
        "schwab.auth.client_from_token_file", fake_client_from_token_file
    )

    first = sc._get_client()
    second = sc._get_client()

    assert first is second
    assert len(calls) == 1  # not re-constructed on the second call


def test_get_client_returns_none_and_sets_retry_cooldown_on_failure(monkeypatch):
    monkeypatch.setattr(sc, "get_settings", lambda: _FakeSettings())

    def fake_client_from_token_file(**kwargs):
        raise FileNotFoundError("no token on disk")

    monkeypatch.setattr(
        "schwab.auth.client_from_token_file", fake_client_from_token_file
    )

    assert sc._get_client() is None
    assert sc._client_init_retry_after is not None

    # A second attempt within the cooldown window must not retry construction.
    calls = []
    monkeypatch.setattr(
        "schwab.auth.client_from_token_file", lambda **kwargs: calls.append(1)
    )
    assert sc._get_client() is None
    assert calls == []


def test_get_quote_maps_a_successful_response(monkeypatch):
    fake_client = _FakeSchwabClient()
    monkeypatch.setattr(sc, "_get_client", lambda: fake_client)
    payload = {
        "AAPL": {
            "quote": {
                "bidPrice": 189.5,
                "askPrice": 189.55,
                "bidSize": 200,
                "askSize": 150,
                "lastPrice": 189.52,
                "quoteTime": 1_734_000_000_000,
            }
        }
    }
    monkeypatch.setattr(
        fake_client, "get_quote", lambda symbol: _FakeResponse(200, payload), raising=False
    )

    quote = sc.get_quote("AAPL")

    assert quote is not None
    assert quote.symbol == "AAPL"
    assert quote.bid_price == 189.5
    assert quote.ask_price == 189.55
    assert quote.bid_size == 200
    assert quote.ask_size == 150
    assert quote.last_trade_price == 189.52
    assert quote.updated_at == datetime.fromtimestamp(1_734_000_000_000 / 1000, tz=UTC)


def test_get_quote_returns_none_on_non_200(monkeypatch):
    fake_client = _FakeSchwabClient()
    monkeypatch.setattr(sc, "_get_client", lambda: fake_client)
    monkeypatch.setattr(
        fake_client,
        "get_quote",
        lambda symbol: _FakeResponse(401, text="unauthorized"),
        raising=False,
    )

    assert sc.get_quote("AAPL") is None


def test_get_quote_returns_none_when_client_unavailable(monkeypatch):
    monkeypatch.setattr(sc, "_get_client", lambda: None)

    assert sc.get_quote("AAPL") is None


def test_get_quote_returns_none_and_does_not_raise_on_unexpected_response_shape(monkeypatch):
    fake_client = _FakeSchwabClient()
    monkeypatch.setattr(sc, "_get_client", lambda: fake_client)
    monkeypatch.setattr(
        fake_client, "get_quote", lambda symbol: _FakeResponse(200, {}), raising=False
    )

    assert sc.get_quote("AAPL") is None


def test_get_5min_historicals_maps_candles(monkeypatch):
    fake_client = _FakeSchwabClient()
    monkeypatch.setattr(sc, "_get_client", lambda: fake_client)
    payload = {
        "symbol": "AAPL",
        "empty": False,
        "candles": [
            {
                "datetime": 1_734_000_000_000,
                "open": 189.0,
                "high": 189.6,
                "low": 188.9,
                "close": 189.5,
                "volume": 12345,
            }
        ],
    }
    monkeypatch.setattr(
        fake_client,
        "get_price_history_every_five_minutes",
        lambda symbol, **kwargs: _FakeResponse(200, payload),
        raising=False,
    )

    bars = sc.get_5min_historicals(
        "AAPL", start_datetime=datetime.now(UTC), end_datetime=datetime.now(UTC)
    )

    assert len(bars) == 1
    bar = bars[0]
    assert bar.symbol == "AAPL"
    assert bar.open == 189.0
    assert bar.close == 189.5
    assert bar.volume == 12345
    assert bar.begins_at == datetime.fromtimestamp(1_734_000_000_000 / 1000, tz=UTC)


def test_get_5min_historicals_returns_empty_list_on_non_200(monkeypatch):
    fake_client = _FakeSchwabClient()
    monkeypatch.setattr(sc, "_get_client", lambda: fake_client)
    monkeypatch.setattr(
        fake_client,
        "get_price_history_every_five_minutes",
        lambda symbol, **kwargs: _FakeResponse(500, text="server error"),
        raising=False,
    )

    bars = sc.get_5min_historicals(
        "AAPL", start_datetime=datetime.now(UTC), end_datetime=datetime.now(UTC)
    )

    assert bars == []


def test_get_5min_historicals_returns_empty_list_when_client_unavailable(monkeypatch):
    monkeypatch.setattr(sc, "_get_client", lambda: None)

    bars = sc.get_5min_historicals(
        "AAPL", start_datetime=datetime.now(UTC), end_datetime=datetime.now(UTC)
    )

    assert bars == []


def test_get_5min_historicals_returns_empty_list_on_exception(monkeypatch):
    fake_client = _FakeSchwabClient()
    monkeypatch.setattr(sc, "_get_client", lambda: fake_client)

    def _raise(symbol, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        fake_client, "get_price_history_every_five_minutes", _raise, raising=False
    )

    bars = sc.get_5min_historicals(
        "AAPL", start_datetime=datetime.now(UTC), end_datetime=datetime.now(UTC)
    )

    assert bars == []
