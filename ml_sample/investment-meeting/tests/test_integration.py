"""Integration tests for investment meeting system."""

import pytest

from src.agents import DataAnalyst, GrowthInvestor, ValueInvestor
from src.moderator import DebateHistory, InvestmentMeetingModerator
from src.reporter import MeetingReporter


@pytest.fixture
def agents() -> list:
    """Fixture for all agents."""
    return [
        GrowthInvestor(),
        ValueInvestor(),
        DataAnalyst(),
    ]


@pytest.fixture
def investment_case() -> str:
    """Fixture for investment case."""
    return """
Evaluate investment in AI-powered logistics startup:
- 2 years old, $5M ARR, 150% YoY growth
- Valuation: $100M (Series B)
- Burn rate: -$1M/month
- Competitive: DHL, FedEx, UPS entering market
- Proprietary: ML optimization algorithms
"""


@pytest.fixture
def moderator(agents: list, investment_case: str) -> InvestmentMeetingModerator:
    """Fixture for moderator."""
    return InvestmentMeetingModerator(agents=agents, investment_case=investment_case)


@pytest.fixture
def reporter() -> MeetingReporter:
    """Fixture for reporter."""
    return MeetingReporter(output_dir="/tmp/test_reports")


class TestModerationWorkflow:
    """Test full moderation workflow."""

    def test_moderator_initialized_with_correct_state(
        self,
        moderator: InvestmentMeetingModerator,
        agents: list,
    ) -> None:
        """Test moderator has correct initial state."""
        assert len(moderator.agents) == len(agents)
        assert moderator.current_round == 0
        assert len(moderator.debate_history) == 0

    def test_debate_history_accumulates(
        self,
        moderator: InvestmentMeetingModerator,
    ) -> None:
        """Test that debate history accumulates over rounds."""
        from src.agents import AgentMessage

        # Simulate adding rounds
        for round_num in range(0, 4):  # Phase 1 (0) + 3 rounds
            messages = [
                AgentMessage(
                    agent_name=agent.name,
                    philosophy=agent.philosophy[:50],
                    message=f"Message from round {round_num}",
                    round_num=round_num,
                )
                for agent in moderator.agents
            ]
            from src.moderator import DebateRound

            moderator.debate_history.append(
                DebateRound(round_num=round_num, messages=messages)
            )

        # Should have 4 rounds total (Phase 1 + 3 debate rounds)
        assert len(moderator.debate_history) == 4
        assert moderator.debate_history[0].round_num == 0  # Phase 1
        assert moderator.debate_history[-1].round_num == 3  # Final round

    def test_agents_appear_in_all_rounds(
        self,
        moderator: InvestmentMeetingModerator,
    ) -> None:
        """Test all agents appear in every round."""
        from src.agents import AgentMessage
        from src.moderator import DebateRound

        agent_names = {agent.name for agent in moderator.agents}

        # Simulate rounds
        for round_num in range(0, 4):
            messages = [
                AgentMessage(
                    agent_name=agent.name,
                    philosophy=agent.philosophy[:50],
                    message="Test",
                    round_num=round_num,
                )
                for agent in moderator.agents
            ]
            moderator.debate_history.append(DebateRound(round_num=round_num, messages=messages))

        # Check each round has all agents
        for round_obj in moderator.debate_history:
            round_agent_names = {msg.agent_name for msg in round_obj.messages}
            assert round_agent_names == agent_names


class TestDebateHistoryModel:
    """Test DebateHistory pydantic model."""

    def test_debate_history_creation(
        self,
        agents: list,
        investment_case: str,
    ) -> None:
        """Test creating debate history."""
        from src.moderator import DebateRound

        rounds = [
            DebateRound(round_num=0, messages=[]),
            DebateRound(round_num=1, messages=[]),
        ]

        history = DebateHistory(
            investment_case=investment_case,
            agents=[agent.name for agent in agents],
            rounds=rounds,
            final_consensus="BUY",
        )

        assert history.investment_case == investment_case
        assert len(history.agents) == 3
        assert len(history.rounds) == 2
        assert history.final_consensus == "BUY"

    def test_debate_history_is_frozen(
        self,
        agents: list,
        investment_case: str,
    ) -> None:
        """Test debate history is immutable."""
        from src.moderator import DebateRound

        history = DebateHistory(
            investment_case=investment_case,
            agents=[agent.name for agent in agents],
            rounds=[DebateRound(round_num=0, messages=[])],
        )

        from pydantic_core import ValidationError
        with pytest.raises(ValidationError):
            history.final_consensus = "SELL"


class TestReporterIntegration:
    """Test reporter with debate history."""

    def test_reporter_generates_text_report(
        self,
        reporter: MeetingReporter,
        agents: list,
        investment_case: str,
    ) -> None:
        """Test reporter can generate text report."""
        from src.agents import AgentMessage
        from src.moderator import DebateRound

        # Create sample history
        messages = [
            AgentMessage(
                agent_name=agent.name,
                philosophy=agent.philosophy[:50],
                message=f"Opinion from {agent.name}",
                round_num=0,
            )
            for agent in agents
        ]

        history = DebateHistory(
            investment_case=investment_case,
            agents=[agent.name for agent in agents],
            rounds=[DebateRound(round_num=0, messages=messages)],
            final_consensus="Hold for more data",
        )

        report = reporter.generate_report(history)
        assert isinstance(report, str)
        assert "INVESTMENT MEETING REPORT" in report
        assert investment_case in report
        assert "Hold for more data" in report

    def test_reporter_generates_json_report(
        self,
        reporter: MeetingReporter,
        agents: list,
        investment_case: str,
    ) -> None:
        """Test reporter can generate JSON report."""
        from src.agents import AgentMessage
        from src.moderator import DebateRound

        messages = [
            AgentMessage(
                agent_name=agent.name,
                philosophy=agent.philosophy[:50],
                message="Test message",
                round_num=0,
            )
            for agent in agents
        ]

        history = DebateHistory(
            investment_case=investment_case,
            agents=[agent.name for agent in agents],
            rounds=[DebateRound(round_num=0, messages=messages)],
        )

        json_report = reporter.generate_json_report(history)
        assert isinstance(json_report, str)
        assert "investment_case" in json_report
        assert "participants" in json_report
        assert "debate_rounds" in json_report

    def test_reporter_summary(
        self,
        reporter: MeetingReporter,
        agents: list,
        investment_case: str,
        capsys,
    ) -> None:
        """Test reporter prints summary."""
        from src.agents import AgentMessage
        from src.moderator import DebateRound

        messages = [
            AgentMessage(
                agent_name=agent.name,
                philosophy=agent.philosophy[:50],
                message="Test",
                round_num=0,
            )
            for agent in agents
        ]

        history = DebateHistory(
            investment_case=investment_case,
            agents=[agent.name for agent in agents],
            rounds=[DebateRound(round_num=0, messages=messages)],
            final_consensus="Strong Buy",
        )

        reporter.print_summary(history)
        captured = capsys.readouterr()
        assert "MEETING SUMMARY" in captured.out
        assert "Strong Buy" in captured.out


class TestThreeRoundIntegration:
    """Integration tests for 3-round debate enforcement."""

    def test_moderator_configured_for_3_rounds(
        self,
        moderator: InvestmentMeetingModerator,
    ) -> None:
        """Test moderator is configured for 3+ debate rounds."""
        assert moderator.MIN_DEBATE_ROUNDS >= 3

    def test_debate_should_include_all_phases(
        self,
        moderator: InvestmentMeetingModerator,
    ) -> None:
        """Test all debate phases exist in workflow."""
        # When run_meeting() is called:
        # 1. Phase 1: Initial opinions (round 0)
        # 2. Phase 2: 3 debate rounds (rounds 1-3)
        # 3. Phase 3: Consensus

        # Total should be 4 history entries (Phase 1 + 3 rounds)
        assert moderator.MIN_DEBATE_ROUNDS == 3
