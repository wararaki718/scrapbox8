"""Moderator for investment meeting with enforced multi-round debate."""

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict

from .agents import AgentMessage, InvestmentAgent


class DebateRound(BaseModel):
    """Model representing a single round of debate."""

    model_config = ConfigDict(frozen=True)

    round_num: int
    messages: list[AgentMessage]


class DebateHistory(BaseModel):
    """Complete history of investment meeting debate."""

    model_config = ConfigDict(frozen=True)

    investment_case: str
    agents: list[str]
    rounds: list[DebateRound]
    final_consensus: str | None = None


class InvestmentMeetingModerator:
    """Moderates investment discussion with enforced debate rounds."""

    MIN_DEBATE_ROUNDS = 3

    def __init__(
        self,
        agents: list[InvestmentAgent],
        investment_case: str,
        tools: list[Any] | None = None,
    ) -> None:
        """Initialize moderator.

        Args:
            agents: List of investment agents to moderate
            investment_case: Description of investment opportunity
            tools: Optional tool definitions for agents to use
        """
        self.agents = agents
        self.investment_case = investment_case
        self.tools = tools
        self.debate_history: list[DebateRound] = []
        self.current_round = 0

    async def run_meeting(self) -> DebateHistory:
        """Run complete investment meeting with enforced 3+ rounds.

        Returns:
            Complete debate history
        """
        # Phase 1: Initial opinions
        print("\n" + "=" * 80)
        print("フェーズ1: 初期意見の提示")
        print("=" * 80)
        await self._phase1_initial_opinions()

        # Phase 2: Enforced debate rounds (minimum 3)
        print("\n" + "=" * 80)
        print(f"フェーズ2: 3往復のディベート ({self.MIN_DEBATE_ROUNDS} ラウンド)")
        print("=" * 80)
        for round_num in range(1, self.MIN_DEBATE_ROUNDS + 1):
            print(f"\n--- ラウンド {round_num} / {self.MIN_DEBATE_ROUNDS} ---")
            await self._debate_round(round_num)

        # Phase 3: Consensus and summary
        print("\n" + "=" * 80)
        print("フェーズ3: 合意形成とリスク評価")
        print("=" * 80)
        final_consensus = await self._phase3_consensus()

        return DebateHistory(
            investment_case=self.investment_case,
            agents=[agent.name for agent in self.agents],
            rounds=self.debate_history,
            final_consensus=final_consensus,
        )

    async def _phase1_initial_opinions(self) -> None:
        """Phase 1: Each agent provides initial investment opinion."""
        initial_prompt = f"""
You are evaluating the following investment opportunity:

{self.investment_case}

Provide your initial investment recommendation (BUY, HOLD, or SELL) with
2-3 key points supporting your view. Be concise but assertive.
"""

        tasks = [
            self._get_agent_response(agent, initial_prompt, phase="Phase 1")
            for agent in self.agents
        ]
        messages = await asyncio.gather(*tasks)

        # Store Phase 1 as Round 0
        round_0 = DebateRound(round_num=0, messages=messages)
        self.debate_history.append(round_0)

    async def _debate_round(self, round_num: int) -> None:
        """Execute single debate round with challenge/counter-argument.

        Args:
            round_num: Current round number (1-3+)
        """
        self.current_round = round_num

        # Build context from previous round for this debate round
        context_summary = self._build_context_for_round(round_num)

        # Each agent must challenge and provide counter-argument
        challenge_prompt = f"""
In this round {round_num} of our investment debate, you must challenge the
previous arguments and defend your position.

CRITICAL RULES FOR THIS ROUND:
1. You MUST identify at least 1 logical weakness or blind spot in the
   previous speakers' arguments
2. You MUST articulate why your investment philosophy is more appropriate
   for this situation
3. Do NOT simply agree with others - defend your viewpoint vigorously
4. Be specific: cite concrete concerns or opportunities

Previous discussion context:
{context_summary}

Now provide your counter-argument and challenge (3-4 sentences):
"""

        tasks = [
            self._get_agent_response(
                agent,
                challenge_prompt,
                phase=f"Debate Round {round_num}",
            )
            for agent in self.agents
        ]
        messages = await asyncio.gather(*tasks)

        # Store debate round
        debate_round = DebateRound(round_num=round_num, messages=messages)
        self.debate_history.append(debate_round)

        # Print for visibility
        for msg in messages:
            print(f"\n[{msg.agent_name}] ラウンド {round_num}:")
            print(msg.message[:300] + "..." if len(msg.message) > 300 else msg.message)

    async def _phase3_consensus(self) -> str:
        """Phase 3: Synthesize key risks and investment thesis.

        Returns:
            Final consensus summary
        """
        debate_summary = self._build_final_summary()

        consensus_prompt = f"""
After {self.MIN_DEBATE_ROUNDS} rounds of debate among our expert panel
(Growth Investor, Value Investor, Data Analyst), synthesize the key
investment risks and opportunities.

Debate summary:
{debate_summary}

Provide a balanced assessment of:
1. Key UPSIDE opportunities (acknowledged by growth/bull perspective)
2. Key DOWNSIDE risks (acknowledged by value/bear perspective)
3. Critical DATA POINTS needed to reduce uncertainty
4. Investment decision: BUY, HOLD, SELL (with one sentence reasoning)

Keep response under 200 words.
"""

        # Use one agent to synthesize (neutral data analyst preferred)
        analyst = next(
            (a for a in self.agents if a.name == "Data Analyst"),
            self.agents[0],
        )
        response = await self._get_agent_response(
            analyst,
            consensus_prompt,
            phase="Consensus",
        )

        print(f"\n[最終合意]\n{response.message}")
        return response.message

    async def _get_agent_response(
        self,
        agent: InvestmentAgent,
        prompt: str,
        phase: str,
    ) -> AgentMessage:
        """Get response from agent and wrap in message object.

        Args:
            agent: Investment agent
            prompt: Prompt to send to agent
            phase: Current phase for logging

        Returns:
            Agent message object
        """
        context = self._build_context_history(agent.name)
        response = await agent.generate_response(prompt, context=context, tools=self.tools)

        return AgentMessage(
            agent_name=agent.name,
            philosophy=agent.philosophy[:100],
            message=response,
            round_num=self.current_round,
        )

    def _build_context_for_round(self, round_num: int) -> str:
        """Build context string from previous rounds.

        Args:
            round_num: Current round

        Returns:
            Context string
        """
        if round_num == 1:
            # For round 1, use Phase 1 (initial opinions)
            initial_round = self.debate_history[0]
            context_parts = [
                f"[{msg.agent_name}]: {msg.message[:200]}" for msg in initial_round.messages
            ]
            return "Initial positions:\n" + "\n".join(context_parts)

        # For rounds 2+, use previous debate round
        prev_round = self.debate_history[-1]
        context_parts = [f"[{msg.agent_name}]: {msg.message[:200]}" for msg in prev_round.messages]
        return "Previous round arguments:\n" + "\n".join(context_parts)

    def _build_context_history(self, agent_name: str) -> str:
        """Build conversation history for context.

        Args:
            agent_name: Name of agent

        Returns:
            History string
        """
        history_items = []
        for round_obj in self.debate_history:
            for msg in round_obj.messages:
                if msg.agent_name != agent_name:
                    history_items.append(
                        f"[Round {msg.round_num} - {msg.agent_name}]:\n{msg.message[:150]}"
                    )

        return "\n---\n".join(history_items[-6:])  # Last 6 messages for context

    def _build_final_summary(self) -> str:
        """Build summary of all debate rounds.

        Returns:
            Formatted debate summary
        """
        summary_parts = ["INVESTMENT CASE:", f"'{self.investment_case}'", ""]

        for round_obj in self.debate_history:
            if round_obj.round_num == 0:
                summary_parts.append("INITIAL POSITIONS:")
            else:
                summary_parts.append(f"DEBATE ROUND {round_obj.round_num}:")

            for msg in round_obj.messages:
                summary_parts.append(f"  [{msg.agent_name}]: {msg.message[:200]}...")

        return "\n".join(summary_parts)
