"""Main entry point for investment meeting simulator."""

import asyncio
import os

from dotenv import load_dotenv

from .agents import DataAnalyst, GrowthInvestor, ValueInvestor
from .moderator import InvestmentMeetingModerator
from .reporter import MeetingReporter

# Load environment variables
load_dotenv()


async def main(
    api_key: str | None = None,
    investment_case: str | None = None,
) -> None:
    """Run investment meeting simulator.

    Args:
        api_key: Google API key for Gemini
        investment_case: Investment opportunity description
    """
    # Setup Gemini API
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. Please set it or pass api_key parameter."
        )

    # google.genai client will use GEMINI_API_KEY environment variable automatically
    # or we can pass it directly when creating Client()

    # Define investment case if not provided
    if investment_case is None:
        investment_case = """
Evaluate investment opportunity: TechCloud Inc.
- Early-stage cloud infrastructure startup (3 years old)
- Revenue: $10M ARR, growing 200% YoY
- Current burn rate: -$2M/month (pre-profitability)
- Funded at $500M valuation (Series C round)
- Market: Cloud infrastructure (estimated $100B+ TAM)
- Competitive landscape: Competing with AWS, Google Cloud, Azure

Key considerations:
- Strong technical team with ex-Google/AWS engineers
- Proprietary ML-based optimization algorithms
- Customer concentration: 3 customers = 60% of revenue
- Operating margin: -120% (burning cash fast)
- Market maturity: AWS/GCP/Azure have dominant positions

Investment decision needed: Allocate $50M venture capital?
"""

    print("\n" + "=" * 80)
    print("INVESTMENT MEETING SIMULATOR")
    print("Using Gemini 2.5 Flash Lite for multi-agent debate")
    print("=" * 80)
    print(f"\nINVESTMENT CASE:\n{investment_case}")

    # Initialize agents with different philosophies
    growth_investor = GrowthInvestor()
    value_investor = ValueInvestor()
    data_analyst = DataAnalyst()

    agents = [growth_investor, value_investor, data_analyst]

    # Run moderated meeting (enforces 3 debate rounds)
    moderator = InvestmentMeetingModerator(
        agents=agents,
        investment_case=investment_case,
    )

    print("\n[会議を開始中...]")
    debate_history = await moderator.run_meeting()

    # Generate and save reports
    reporter = MeetingReporter(output_dir="./reports")
    text_path, json_path = reporter.save_report(
        debate_history,
        filename_base="investment_decision",
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
            filename_base="investment_decision",
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
            filename_base="investment_decision",
        )
        print(f"✅ 強化済みJSON: {enhanced_path}")
    except Exception as e:
        print(f"⚠️  JSON翻訳エラー: {e}")


if __name__ == "__main__":
    asyncio.run(main())
