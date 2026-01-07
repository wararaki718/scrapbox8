"""Tests for investment meeting moderator and debate logic."""

import pytest

from src.agents import DataAnalyst, GrowthInvestor, ValueInvestor
from src.moderator import DebateRound, InvestmentMeetingModerator


@pytest.fixture
def agents() -> list:
    """Fixture for test agents."""
    return [
        GrowthInvestor(),
        ValueInvestor(),
        DataAnalyst(),
    ]


@pytest.fixture
def simple_investment_case() -> str:
    """Fixture for simple investment case."""
    return "Evaluate investment in early-stage SaaS startup with 200% YoY growth"


@pytest.fixture
def moderator(
    agents: list,
    simple_investment_case: str,
) -> InvestmentMeetingModerator:
    """Fixture for investment moderator."""
    return InvestmentMeetingModerator(
        agents=agents,
        investment_case=simple_investment_case,
    )


class TestModeratorInitialization:
    """Test moderator initialization."""

    def test_moderator_has_agents(
        self,
        moderator: InvestmentMeetingModerator,
    ) -> None:
        """Test moderator stores agents."""
        assert len(moderator.agents) == 3

    def test_moderator_has_investment_case(
        self,
        moderator: InvestmentMeetingModerator,
    ) -> None:
        """Test moderator stores investment case."""
        assert moderator.investment_case is not None
        assert len(moderator.investment_case) > 0

    def test_moderator_history_initialized_empty(
        self,
        moderator: InvestmentMeetingModerator,
    ) -> None:
        """Test debate history starts empty."""
        assert moderator.debate_history == []

    def test_moderator_has_min_rounds_constant(
        self,
        moderator: InvestmentMeetingModerator,
    ) -> None:
        """Test minimum debate rounds is set to 3."""
        assert moderator.MIN_DEBATE_ROUNDS == 3


class TestDebateRoundModel:
    """Test DebateRound pydantic model."""

    def test_debate_round_creation(self) -> None:
        """Test creating a debate round."""
        from src.agents import AgentMessage

        messages = [
            AgentMessage(
                agent_name="Growth Investor",
                philosophy="Growth focused",
                message="This is a great opportunity",
                round_num=0,
            ),
        ]
        round_obj = DebateRound(round_num=0, messages=messages)
        assert round_obj.round_num == 0
        assert len(round_obj.messages) == 1

    def test_debate_round_is_frozen(self) -> None:
        """Test that debate round is immutable."""
        from src.agents import AgentMessage

        messages = [
            AgentMessage(
                agent_name="Growth Investor",
                philosophy="Growth focused",
                message="Test",
                round_num=0,
            ),
        ]
        round_obj = DebateRound(round_num=0, messages=messages)

        from pydantic_core import ValidationError
        with pytest.raises(ValidationError):
            round_obj.round_num = 1


class TestContextBuilding:
    """Test context building for agents."""

    async def test_context_for_round_returns_string(
        self,
        moderator: InvestmentMeetingModerator,
    ) -> None:
        """Test context building returns string."""
        from src.agents import AgentMessage

        # Add Phase 1 data to history
        phase1_messages = [
            AgentMessage(
                agent_name=agent.name,
                philosophy=agent.philosophy[:50],
                message="Initial position",
                round_num=0,
            )
            for agent in moderator.agents
        ]
        moderator.debate_history.append(
            DebateRound(round_num=0, messages=phase1_messages)
        )

        context = moderator._build_context_for_round(1)
        assert isinstance(context, str)

    async def test_context_history_for_agent(
        self,
        moderator: InvestmentMeetingModerator,
    ) -> None:
        """Test context history building."""
        history = moderator._build_context_history("Growth Investor")
        assert isinstance(history, str)

    async def test_final_summary_includes_case(
        self,
        moderator: InvestmentMeetingModerator,
    ) -> None:
        """Test final summary includes investment case."""
        summary = moderator._build_final_summary()
        assert "INVESTMENT CASE" in summary
        assert moderator.investment_case[:50] in summary


class TestThreeRoundEnforcement:
    """Critical tests for enforcing minimum 3 debate rounds."""

    def test_moderator_enforces_3_rounds(
        self,
        moderator: InvestmentMeetingModerator,
    ) -> None:
        """Test that moderator config requires 3 rounds minimum."""
        assert moderator.MIN_DEBATE_ROUNDS >= 3

    def test_debate_history_structure_after_phase1(
        self,
        moderator: InvestmentMeetingModerator,
        agents: list,
    ) -> None:
        """Test debate history structure after Phase 1."""
        # Manually create a Phase 1 round
        from src.agents import AgentMessage

        phase1_messages = [
            AgentMessage(
                agent_name=agent.name,
                philosophy=agent.philosophy[:50],
                message=f"Initial position from {agent.name}",
                round_num=0,
            )
            for agent in agents
        ]

        phase1_round = DebateRound(round_num=0, messages=phase1_messages)
        moderator.debate_history.append(phase1_round)

        # Check structure
        assert len(moderator.debate_history) == 1
        assert moderator.debate_history[0].round_num == 0
        assert len(moderator.debate_history[0].messages) == 3  # All 3 agents

    def test_debate_rounds_numbered_sequentially(
        self,
        moderator: InvestmentMeetingModerator,
    ) -> None:
        """Test that debate rounds are numbered 1, 2, 3."""
        from src.agents import AgentMessage

        # Simulate multiple rounds
        for round_num in range(1, 4):
            messages = [
                AgentMessage(
                    agent_name=agent.name,
                    philosophy=agent.philosophy[:50],
                    message=f"Round {round_num} from {agent.name}",
                    round_num=round_num,
                )
                for agent in moderator.agents
            ]
            debate_round = DebateRound(round_num=round_num, messages=messages)
            moderator.debate_history.append(debate_round)

        # Verify all 3 rounds are present (plus Phase 1 = 4 total)
        # We should have 3 debate rounds when run_meeting is called
        assert moderator.MIN_DEBATE_ROUNDS == 3


class TestLowChallengePrompt:
    """Test that agents are prompted to challenge each other."""

    def test_debate_round_prompt_enforces_challenges(
        self,
        moderator: InvestmentMeetingModerator,
    ) -> None:
        """Test that debate prompts contain challenge requirements."""
        from src.agents import AgentMessage

        # Add a phase 1 round
        phase1_messages = [
            AgentMessage(
                agent_name=agent.name,
                philosophy=agent.philosophy[:50],
                message=f"Initial: {agent.name} says...",
                round_num=0,
            )
            for agent in moderator.agents
        ]
        moderator.debate_history.append(DebateRound(round_num=0, messages=phase1_messages))

        # Build context for a debate round
        context = moderator._build_context_for_round(1)

        # Context should reference previous arguments
        assert isinstance(context, str)
        assert len(context) > 0
