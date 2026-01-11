"""Investment agent definitions with distinct philosophies."""

from typing import Any

import google.genai as genai
from pydantic import BaseModel, ConfigDict


class AgentMessage(BaseModel):
    """Model representing a message from an agent."""

    model_config = ConfigDict(frozen=True)

    agent_name: str
    philosophy: str
    message: str
    round_num: int


class InvestmentAgent:
    """Base class for investment decision agents."""

    def __init__(
        self,
        name: str,
        philosophy: str,
        model_name: str = "gemini-2.5-flash-lite",
    ) -> None:
        """Initialize investment agent.

        Args:
            name: Agent identifier
            philosophy: Investment philosophy description
            model_name: Gemini model to use
        """
        self.name = name
        self.philosophy = philosophy
        self.model_name = model_name
        # Using a single client instance, but we will use the async interface
        self.client: Any = genai.Client()
        self.conversation_history: list[str] = []

    def add_to_history(self, message: str) -> None:
        """Add message to conversation history.

        Args:
            message: Message content
        """
        self.conversation_history.append(message)

    def get_system_prompt(self) -> str:
        """Get system prompt for this agent.

        Returns:
            System prompt string
        """
        base_prompt = f"""You are an investment advisor with the following philosophy:
{self.philosophy}

Your role is to provide investment analysis based on your specific philosophy.
Be assertive about your viewpoint and don't hesitate to challenge others' logic.
Focus on your core principles when analyzing investment decisions.

CRITICAL: You MUST use the provided MCP tools to fetch the latest stock price and news
for the target ticker before forming your opinion. Your arguments MUST be based on
FACTS and DATA retrieved from these tools, not on feelings or general knowledge.
Point out specific weaknesses in others' logic using the retrieved data.
"""
        return base_prompt

    async def generate_response(
        self,
        prompt: str,
        context: str | None = None,
        tools: list[Any] | None = None,
    ) -> str:
        """Generate response using Gemini model.

        Args:
            prompt: Input prompt for the agent
            context: Optional context from other agents
            tools: Optional tool definitions for Gemini

        Returns:
            Generated response
        """
        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\n{prompt}"

        response = await self._call_model(full_prompt, tools)
        self.add_to_history(response)
        return response

    async def _call_model(self, prompt: str, tools: list[Any] | None = None) -> str:
        """Call Gemini model with tool support.

        Args:
            prompt: Input prompt
            tools: Optional tools

        Returns:
            Model response
        """
        system_prompt = self.get_system_prompt()
        try:
            # Use the async client (aio) to support async tool functions
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=f"{system_prompt}\n\n{prompt}",
                config={
                    "tools": tools,
                    "automatic_function_calling": {"disable": False},
                    "max_output_tokens": 1000,
                    "temperature": 0.2,
                },
            )
            return response.text if response.text else "No response generated"
        except Exception as e:
            return f"Error generating response: {str(e)}"


class GrowthInvestor(InvestmentAgent):
    """Aggressive growth investor focusing on future potential."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
    ) -> None:
        """Initialize growth investor agent."""
        philosophy = """
You are an aggressive Growth Investor who believes in:
- Disruptive innovation and market disruption potential
- Future cash flow projections and exponential growth
- Market share capture and winner-takes-all dynamics
- Embracing volatility as opportunity
- Long-term value creation over short-term safety

You prioritize:
1. TAM (Total Addressable Market) expansion
2. Unit economics and scalability
3. Competitive moats through innovation
4. Management's vision for the future

When challenged, defend your bullish thesis vigorously and highlight growth
opportunities that others might be missing. Point out how conservative
valuations may underestimate disruption.
"""
        super().__init__(
            name="Growth Investor",
            philosophy=philosophy,
            model_name=model_name,
        )


class ValueInvestor(InvestmentAgent):
    """Conservative value investor focusing on safety."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
    ) -> None:
        """Initialize value investor agent."""
        philosophy = """
You are a conservative Value Investor who believes in:
- Margin of Safety: only invest at significant discounts to intrinsic value
- Balance sheet strength (low debt, high liquidity)
- PBR (Price-to-Book Ratio) and PER (Price-to-Earnings) analysis
- Proven cash generation over speculative potential
- Capital preservation as primary goal

You prioritize:
1. Financial stability and balance sheet health
2. Free cash flow generation
3. Valuation discipline (no overpaying)
4. Downside protection

When challenged, emphasize the risks others are overlooking and the dangers
of overpaying for uncertain growth. Point out red flags in financial health
and unsustainable valuations.
"""
        super().__init__(
            name="Value Investor",
            philosophy=philosophy,
            model_name=model_name,
        )


class DataAnalyst(InvestmentAgent):
    """Data-driven analyst with objective, critical perspective."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
    ) -> None:
        """Initialize data analyst agent."""
        philosophy = """
You are a Data-Driven Analyst who believes in:
- Macro-economic indicators and market trends
- Comparative analysis against peer companies
- Quantitative metrics from recent financial reports
- Statistical rigor and evidence-based reasoning
- Maintaining objectivity and neutrality

You prioritize:
1. Latest financial data and industry benchmarks
2. Competitive positioning analysis
3. Macro headwinds and tailwinds
4. Risk/reward ratio calculation

When challenged, back your claims with specific data points, recent figures,
and industry comparisons. Point out logical fallacies in others' reasoning
and demand evidence for bullish or bearish claims.
"""
        super().__init__(
            name="Data Analyst",
            philosophy=philosophy,
            model_name=model_name,
        )
