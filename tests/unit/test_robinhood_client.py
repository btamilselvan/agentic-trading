from datetime import UTC, datetime

from agentic_trading.market_data import robinhood_client as rhc


def _bypass_login(monkeypatch):
    monkeypatch.setattr(rhc, "ensure_login", lambda: None)


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
