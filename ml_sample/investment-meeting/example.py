"""Quick start example for investment meeting simulator."""

import asyncio
import os

import google.genai as genai
from dotenv import load_dotenv

from src.agents import DataAnalyst, GrowthInvestor, ValueInvestor
from src.moderator import InvestmentMeetingModerator
from src.reporter import MeetingReporter


async def run_quick_example() -> None:
    """Run a quick example investment meeting."""
    # Setup
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("⚠️  GOOGLE_API_KEY not found in environment.")
        print("Please set GOOGLE_API_KEY or create a .env file:")
        print("  echo 'GOOGLE_API_KEY=your_key' > .env")
        return

    # google.genai client will use GOOGLE_API_KEY environment variable automatically

    # Define investment case
    investment_case = """
Evaluate: TechAI Startup
- Founded: 2022
- Current valuation: $200M (Series B)
- Revenue: $8M ARR, 180% YoY growth
- Burn rate: $1.5M/month
- Team: ex-OpenAI, ex-DeepMind engineers
- Product: AI-powered code generation tool
- Competition: GitHub Copilot, JetBrains AI Assistant
- Key risks: High burn rate, regulatory uncertainty, commoditization risk
- Key opportunities: AI market growth (200%+ CAGR), strong developer adoption

Investment question: Should we lead a $100M Series C round at $1B valuation?
"""

    print("\n" + "=" * 80)
    print("INVESTMENT MEETING EXAMPLE")
    print("=" * 80)
    print(f"Case: TechAI Startup\nValuation: $1B (Series C)")
    print("Participants: Growth Investor, Value Investor, Data Analyst")
    print("="* 80)

    # Initialize agents
    agents = [
        GrowthInvestor(),
        ValueInvestor(),
        DataAnalyst(),
    ]

    # Create and run moderator (3 debate rounds enforced)
    moderator = InvestmentMeetingModerator(
        agents=agents,
        investment_case=investment_case,
    )

    print("\n[Running investment meeting with 3-round debate...]")
    debate_history = await moderator.run_meeting()

    # Generate reports
    reporter = MeetingReporter(output_dir="./reports")
    text_path, json_path = reporter.save_report(
        debate_history,
        filename_base="techai_investment",
    )

    reporter.print_summary(debate_history)

    print(f"\n✅ Meeting complete!")
    print(f"   Text report: {text_path}")
    print(f"   JSON report: {json_path}")


if __name__ == "__main__":
    asyncio.run(run_quick_example())
