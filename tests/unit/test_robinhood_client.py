from datetime import UTC, datetime

import pytest

from agentic_trading.market_data import robinhood_client as rhc


def _bypass_login(monkeypatch):
    monkeypatch.setattr(rhc, "ensure_login", lambda: None)


@pytest.fixture(autouse=True)
def _reset_login_state():
    """ensure_login() caches success/failure in module-level globals -- isolate tests
    from each other and from whatever real login state a prior test in the suite left."""
    rhc._logged_in = False
    rhc._login_retry_after = None
    yield
    rhc._logged_in = False
    rhc._login_retry_after = None


def test_ensure_login_marks_logged_in_on_success(monkeypatch):
    monkeypatch.setattr(
        rhc.rh, "login", lambda **kwargs: {"access_token": "tok", "token_type": "Bearer"}
    )

    rhc.ensure_login()

    assert rhc._logged_in is True


def test_ensure_login_is_a_no_op_once_already_logged_in(monkeypatch):
    rhc._logged_in = True

    def _fail_if_called(**kwargs):
        raise AssertionError("should not be called")

    monkeypatch.setattr(rhc.rh, "login", _fail_if_called)

    rhc.ensure_login()  # must not raise / must not call rh.login again


def test_ensure_login_does_not_raise_and_does_not_cache_when_robin_stocks_returns_none(
    monkeypatch,
):
    """Mirrors the real robin_stocks failure mode: login() prints 'Login failed' and
    returns None instead of raising (e.g. stale pickle + unapproved device verification).
    This must not raise -- none of this module's market-data calls are login_required,
    so an unauthenticated session still works and callers shouldn't be blocked."""
    monkeypatch.setattr(rhc.rh, "login", lambda **kwargs: None)

    rhc.ensure_login()  # must not raise

    assert rhc._logged_in is False


def test_ensure_login_does_not_raise_when_response_is_missing_access_token(monkeypatch):
    monkeypatch.setattr(rhc.rh, "login", lambda **kwargs: {"detail": "verification pending"})

    rhc.ensure_login()  # must not raise

    assert rhc._logged_in is False


def test_ensure_login_blocks_immediate_retry_with_a_cooldown_after_a_failure(monkeypatch):
    """A failed login must not be retried on the very next call -- each full login mints
    a new device_token, so hammering it looks like a burst of new-device logins and risks
    Robinhood throttling the login endpoint itself."""
    calls = []
    monkeypatch.setattr(rhc.rh, "login", lambda **kwargs: calls.append(1) or None)

    rhc.ensure_login()
    assert len(calls) == 1

    rhc.ensure_login()
    assert len(calls) == 1  # rh.login() not called again -- blocked by the cooldown


def test_ensure_login_retries_after_the_cooldown_elapses(monkeypatch):
    calls = []

    def fake_login(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            return None
        return {"access_token": "tok", "token_type": "Bearer"}

    monkeypatch.setattr(rhc.rh, "login", fake_login)

    rhc.ensure_login()
    assert rhc._logged_in is False

    # Simulate the cooldown window having elapsed.
    rhc._login_retry_after = None

    rhc.ensure_login()
    assert rhc._logged_in is True
    assert len(calls) == 2


def test_get_latest_news_picks_the_most_recently_published_story(monkeypatch):
    _bypass_login(monkeypatch)
    monkeypatch.setattr(
        rhc.rh.stocks,
        "get_news",
        lambda symbol: [
            {
                "title": "Older story",
                "summary": "s1",
                "published_at": "2026-08-17T09:00:00Z",
                "source": "A",
            },
            {
                "title": "Newer story",
                "summary": "s2",
                "published_at": "2026-08-19T09:00:00Z",
                "source": "B",
            },
        ],
    )

    news = rhc.get_latest_news("AAPL")

    assert news.title == "Newer story"
    assert news.summary == "s2"
    assert news.published_at == datetime(2026, 8, 19, 9, 0, tzinfo=UTC)
    assert news.source == "B"


def test_get_latest_news_falls_back_to_feed_order_without_timestamps(monkeypatch):
    _bypass_login(monkeypatch)
    monkeypatch.setattr(
        rhc.rh.stocks,
        "get_news",
        lambda symbol: [
            {"title": "First in feed", "summary": None, "published_at": None, "source": None},
            {"title": "Second in feed", "summary": None, "published_at": None, "source": None},
        ],
    )

    news = rhc.get_latest_news("AAPL")

    assert news.title == "First in feed"
    assert news.published_at is None


def test_get_latest_news_skips_rows_without_a_title(monkeypatch):
    _bypass_login(monkeypatch)
    monkeypatch.setattr(
        rhc.rh.stocks,
        "get_news",
        lambda symbol: [
            {"title": "", "summary": None, "published_at": None, "source": None},
            None,
            {
                "title": "Valid story",
                "summary": None,
                "published_at": None,
                "source": None,
            },
        ],
    )

    news = rhc.get_latest_news("AAPL")

    assert news.title == "Valid story"


def test_get_latest_news_none_when_no_stories(monkeypatch):
    _bypass_login(monkeypatch)
    monkeypatch.setattr(rhc.rh.stocks, "get_news", lambda symbol: [])

    assert rhc.get_latest_news("AAPL") is None


def test_get_float_shares_parses_the_fundamentals_float_field(monkeypatch):
    _bypass_login(monkeypatch)
    monkeypatch.setattr(
        rhc.rh.stocks, "get_fundamentals", lambda symbol: [{"float": "15000000.0000"}]
    )

    assert rhc.get_float_shares("GME") == 15_000_000


def test_get_float_shares_none_when_field_missing_or_blank(monkeypatch):
    _bypass_login(monkeypatch)
    monkeypatch.setattr(rhc.rh.stocks, "get_fundamentals", lambda symbol: [{"float": ""}])
    assert rhc.get_float_shares("AAPL") is None


def test_get_float_shares_none_when_fundamentals_unavailable(monkeypatch):
    _bypass_login(monkeypatch)
    monkeypatch.setattr(rhc.rh.stocks, "get_fundamentals", lambda symbol: [None])
    assert rhc.get_float_shares("AAPL") is None
