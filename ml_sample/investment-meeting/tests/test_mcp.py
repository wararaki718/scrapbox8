"""Tests for MCP server tools."""

from src.mcp_server import get_news, get_stock_price


def test_get_stock_price():
    """Test get_stock_price tool with a known ticker."""
    # We use a real ticker but we don't assert exact values as they change
    result = get_stock_price("AAPL")
    assert "Ticker: AAPL" in result
    assert "Current Price:" in result
    assert "Error" not in result


def test_get_news():
    """Test get_news tool with a known ticker."""
    result = get_news("AAPL")
    assert "AAPL" in result
    # It should return news headlines or a "no news found" message
    assert isinstance(result, str)
    assert len(result) > 0


def test_invalid_ticker():
    """Test tools with an invalid ticker."""
    result = get_stock_price("INVALID_TICKER_12345")
    # yfinance usually returns an error or empty data for invalid tickers
    assert "Error" in result or "None" in result or "Ticker: INVALID_TICKER_12345" in result
