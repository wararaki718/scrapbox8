"""Tests for investment meeting simulator."""

import pytest

from src.agents import DataAnalyst, GrowthInvestor, ValueInvestor


@pytest.fixture
def growth_investor() -> GrowthInvestor:
    """Fixture for growth investor agent."""
    return GrowthInvestor()


@pytest.fixture
def value_investor() -> ValueInvestor:
    """Fixture for value investor agent."""
    return ValueInvestor()


@pytest.fixture
def data_analyst() -> DataAnalyst:
    """Fixture for data analyst agent."""
    return DataAnalyst()


class TestAgentInitialization:
    """Test agent initialization and properties."""

    def test_growth_investor_name(self, growth_investor: GrowthInvestor) -> None:
        """Test growth investor has correct name."""
        assert growth_investor.name == "Growth Investor"

    def test_value_investor_name(self, value_investor: ValueInvestor) -> None:
        """Test value investor has correct name."""
        assert value_investor.name == "Value Investor"

    def test_data_analyst_name(self, data_analyst: DataAnalyst) -> None:
        """Test data analyst has correct name."""
        assert data_analyst.name == "Data Analyst"

    def test_agent_has_philosophy(
        self,
        growth_investor: GrowthInvestor,
    ) -> None:
        """Test agent has philosophy string."""
        assert len(growth_investor.philosophy) > 0
        assert "Growth" in growth_investor.philosophy or "innovation" in (
            growth_investor.philosophy.lower()
        )

    def test_agent_conversation_history_initialized(
        self,
        growth_investor: GrowthInvestor,
    ) -> None:
        """Test agent conversation history is initialized as empty list."""
        assert growth_investor.conversation_history == []


class TestAgentHistoryManagement:
    """Test agent conversation history management."""

    def test_add_to_history(self, growth_investor: GrowthInvestor) -> None:
        """Test adding message to history."""
        message = "Test message"
        growth_investor.add_to_history(message)
        assert message in growth_investor.conversation_history

    def test_multiple_history_entries(
        self,
        growth_investor: GrowthInvestor,
    ) -> None:
        """Test multiple history entries are stored in order."""
        messages = ["First", "Second", "Third"]
        for msg in messages:
            growth_investor.add_to_history(msg)

        assert growth_investor.conversation_history == messages


class TestSystemPrompts:
    """Test system prompt generation."""

    def test_growth_investor_prompt_contains_philosophy(
        self,
        growth_investor: GrowthInvestor,
    ) -> None:
        """Test system prompt includes investment philosophy."""
        prompt = growth_investor.get_system_prompt()
        assert "Growth Investor" in prompt or "growth" in prompt.lower()
        assert len(prompt) > 100

    def test_value_investor_prompt_contains_philosophy(
        self,
        value_investor: ValueInvestor,
    ) -> None:
        """Test value investor prompt contains relevant keywords."""
        prompt = value_investor.get_system_prompt()
        assert "Value" in prompt or "safety" in prompt.lower()

    def test_data_analyst_prompt_contains_philosophy(
        self,
        data_analyst: DataAnalyst,
    ) -> None:
        """Test data analyst prompt mentions objectivity."""
        prompt = data_analyst.get_system_prompt()
        assert "Data" in prompt or "analysis" in prompt.lower()


class TestAgentDistinctPhilosophies:
    """Test that agents have distinct philosophies."""

    def test_all_agents_have_different_names(
        self,
        growth_investor: GrowthInvestor,
        value_investor: ValueInvestor,
        data_analyst: DataAnalyst,
    ) -> None:
        """Test that agents have unique names."""
        names = {growth_investor.name, value_investor.name, data_analyst.name}
        assert len(names) == 3

    def test_philosophies_differ(
        self,
        growth_investor: GrowthInvestor,
        value_investor: ValueInvestor,
        data_analyst: DataAnalyst,
    ) -> None:
        """Test that agent philosophies are distinct."""
        philosophies = {
            growth_investor.philosophy,
            value_investor.philosophy,
            data_analyst.philosophy,
        }
        # Should have 3 distinct philosophies
        assert len(philosophies) == 3
