"""Main entry point for investment meeting simulator."""

import asyncio
import os
import sys

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from .agents import DataAnalyst, GrowthInvestor, ValueInvestor
from .moderator import InvestmentMeetingModerator
from .reporter import MeetingReporter

# Load environment variables
load_dotenv()


async def main(
    api_key: str | None = None,
    investment_case: str | None = None,
    ticker: str = "TSLA",  # Default ticker for MCP tools
) -> None:
    """Run investment meeting simulator with MCP tools.

    Args:
        api_key: Google API key for Gemini
        investment_case: Investment opportunity description
        ticker: Ticker symbol to analyze
    """
    # Setup Gemini API
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. Please set it or pass api_key parameter."
        )

    # MCP Server configuration
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(os.path.dirname(__file__), "mcp_server.py")],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Define tools for Gemini that call MCP tools
            async def get_stock_price_tool(ticker: str) -> str:
                """Get current stock price and financial metrics for a ticker.

                Args:
                    ticker: The stock symbol (e.g. AAPL, TSLA)
                """
                result = await session.call_tool("get_stock_price", {"ticker": ticker})
                if not result.content:
                    return "No data"
                content = result.content[0]
                if isinstance(content, TextContent):
                    return content.text
                return str(content)

            async def get_news_tool(ticker: str) -> str:
                """Get the latest news for a ticker.

                Args:
                    ticker: The stock symbol (e.g. AAPL, TSLA)
                """
                result = await session.call_tool("get_news", {"ticker": ticker})
                if not result.content:
                    return "No news"
                content = result.content[0]
                if isinstance(content, TextContent):
                    return content.text
                return str(content)

            # Combine tools for agents
            tools = [get_stock_price_tool, get_news_tool]

            if investment_case is None:
                investment_case = f"""
Evaluate investment opportunity: {ticker}
Analyze the latest financial metrics and news using the provided MCP tools.
Consider the growth potential vs current valuation and market sentiment.
"""

            print("\n" + "=" * 80)
            print("INVESTMENT MEETING SIMULATOR (MCP Enabled)")
            print("Using Gemini 2.5 Flash Lite + MCP Tools")
            print("=" * 80)
            print(f"\nINVESTMENT CASE:\n{investment_case}")
            print(f"TARGET TICKER: {ticker}")

            # Initialize agents
            growth_investor = GrowthInvestor()
            value_investor = ValueInvestor()
            data_analyst = DataAnalyst()

            agents = [growth_investor, value_investor, data_analyst]

            # Run moderated meeting
            moderator = InvestmentMeetingModerator(
                agents=agents,
                investment_case=investment_case,
                tools=tools,
            )

            print("\n[会議を開始中... MCP ツールを準備完了]")
            debate_history = await moderator.run_meeting()

            # Generate and save reports
            reporter = MeetingReporter(output_dir="./reports")
            text_path, json_path = reporter.save_report(
                debate_history,
                filename_base=f"investment_decision_{ticker}",
            )

            reporter.print_summary(debate_history)

            print("\n✅ レポートを保存しました:")
            print(f"   テキスト: {text_path}")
            print(f"   JSON: {json_path}")

            # Generate translated reports
            print("\n[日本語翻訳レポートを生成中...]")
            try:
                text_report = text_path.read_text(encoding="utf-8")
                translated_text = await reporter.translate_report_to_japanese(
                    text_report,
                )
                translated_path = reporter.save_translated_report(
                    translated_text,
                    filename_base=f"investment_decision_{ticker}",
                )
                print(f"✅ 翻訳済みテキスト: {translated_path}")
            except Exception as e:
                print(f"⚠️  テキスト翻訳エラー: {e}")

            # Generate enhanced JSON report
            try:
                json_content = json_path.read_text(encoding="utf-8")
                enhanced_json = await reporter.enhance_json_report(json_content)
                enhanced_path = reporter.save_enhanced_json_report(
                    enhanced_json,
                    filename_base=f"investment_decision_{ticker}",
                )
                print(f"✅ 強化済みJSON: {enhanced_path}")
            except Exception as e:
                print(f"⚠️  JSON翻訳エラー: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Investment Meeting Simulator")
    parser.add_argument("--ticker", type=str, default="TSLA", help="Ticker symbol to analyze")
    parser.add_argument("--api-key", type=str, help="Gemini API Key")
    args = parser.parse_args()

    asyncio.run(main(api_key=args.api_key, ticker=args.ticker))
