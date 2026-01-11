"""MCP server for investment meeting simulator."""

import yfinance as yf  # type: ignore
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP
mcp = FastMCP("InvestmentToolbox")


@mcp.tool()
def get_stock_price(ticker: str) -> str:
    """Get the current stock price and key metrics for a given ticker.

    Args:
        ticker: The stock symbol (e.g., 'AAPL', 'MSFT', '7203.T').

    Returns:
        A string containing price, PER, PBR, and market cap.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        per = info.get("trailingPE")
        pbr = info.get("priceToBook")
        mkt_cap = info.get("marketCap")

        result = [
            f"Ticker: {ticker}",
            f"Current Price: {current_price}",
            f"Forward PER: {info.get('forwardPE')}",
            f"Trailing PER: {per}",
            f"PBR: {pbr}",
            f"Market Cap: {mkt_cap}",
            f"52 Week High: {info.get('fiftyTwoWeekHigh')}",
            f"52 Week Low: {info.get('fiftyTwoWeekLow')}",
        ]
        return "\n".join(result)
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"


@mcp.tool()
def get_news(ticker: str) -> str:
    """Get the latest news for a given ticker.

    Args:
        ticker: The stock symbol.

    Returns:
        A formatted string of recent news headlines.
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return f"No news found for {ticker}."

        results = [f"News for {ticker}:"]
        for item in news[:5]:  # Get top 5 news
            title = item.get("title") or "No Title"
            publisher = item.get("publisher") or "Unknown"
            link = item.get("link") or "No Link"
            results.append(f"- {title} ({publisher})\n  {link}")

        return "\n\n".join(results)
    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"


if __name__ == "__main__":
    mcp.run()
