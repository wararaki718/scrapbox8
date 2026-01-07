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
            "GEMINI_API_KEY environment variable not set. "
            "Please set it or pass api_key parameter."
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

    print("\n[Starting moderated investment meeting...]")
    debate_history = await moderator.run_meeting()

    # Generate and save reports
    reporter = MeetingReporter(output_dir="./reports")
    text_path, json_path = reporter.save_report(
        debate_history,
        filename_base="investment_decision",
    )

    reporter.print_summary(debate_history)

    print("\n✅ Reports saved:")
    print(f"   Text: {text_path}")
    print(f"   JSON: {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
