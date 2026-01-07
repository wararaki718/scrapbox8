"""Tests for report generation and translation."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.moderator import AgentMessage, DebateHistory, DebateRound
from src.reporter import MeetingReporter


@pytest.fixture
def sample_debate_history() -> DebateHistory:
    """Create sample debate history for testing."""
    messages_phase1 = [
        AgentMessage(
            agent_name="Growth Investor",
            philosophy="Growth",
            message="This company has high growth potential.",
            round_num=0,
        ),
        AgentMessage(
            agent_name="Value Investor",
            philosophy="Value",
            message="The valuation seems high for current revenue.",
            round_num=0,
        ),
    ]

    round_1 = DebateRound(round_num=0, messages=messages_phase1)

    messages_round_1 = [
        AgentMessage(
            agent_name="Growth Investor",
            philosophy="Growth",
            message="The TAM is large and growing rapidly.",
            round_num=1,
        ),
    ]

    round_2 = DebateRound(round_num=1, messages=messages_round_1)

    return DebateHistory(
        investment_case="Test Investment Case",
        agents=["Growth Investor", "Value Investor", "Data Analyst"],
        rounds=[round_1, round_2],
        final_consensus="Consensus reached after 2 rounds.",
    )


def test_reporter_initialization(tmp_path: Path) -> None:
    """Test reporter initialization."""
    reporter = MeetingReporter(output_dir=str(tmp_path))
    assert reporter.output_dir == tmp_path
    assert tmp_path.exists()


def test_generate_report(sample_debate_history: DebateHistory) -> None:
    """Test text report generation."""
    reporter = MeetingReporter()
    report = reporter.generate_report(sample_debate_history, "Test Case")

    assert "投資会議レポート" in report
    assert "Test Case" in report
    assert "Growth Investor" in report
    assert "Value Investor" in report
    assert "最終合意と リスク評価" in report


def test_generate_json_report(sample_debate_history: DebateHistory) -> None:
    """Test JSON report generation."""
    reporter = MeetingReporter()
    json_str = reporter.generate_json_report(sample_debate_history)

    report_data = json.loads(json_str)

    assert "生成日時" in report_data
    assert "投資案件" in report_data
    assert "参加者" in report_data
    assert "ディベートラウンド" in report_data
    assert "最終合意" in report_data
    assert report_data["投資案件"] == "Test Investment Case"


def test_save_report(
    sample_debate_history: DebateHistory,
    tmp_path: Path,
) -> None:
    """Test saving both text and JSON reports."""
    reporter = MeetingReporter(output_dir=str(tmp_path))
    text_path, json_path = reporter.save_report(sample_debate_history)

    assert text_path.exists()
    assert json_path.exists()
    assert text_path.suffix == ".txt"
    assert json_path.suffix == ".json"

    # Verify content
    text_content = text_path.read_text(encoding="utf-8")
    assert "投資会議レポート" in text_content

    json_content = json_path.read_text(encoding="utf-8")
    data = json.loads(json_content)
    assert "生成日時" in data


def test_print_summary(
    sample_debate_history: DebateHistory,
    capsys: pytest.CaptureFixture,
) -> None:
    """Test console summary output."""
    reporter = MeetingReporter()
    reporter.print_summary(sample_debate_history)

    captured = capsys.readouterr()
    assert "投資会議 - サマリー" in captured.out
    assert "Test Investment Case" in captured.out


def test_save_translated_report(tmp_path: Path) -> None:
    """Test saving translated report."""
    reporter = MeetingReporter(output_dir=str(tmp_path))
    translated_text = "翻訳済みレポート内容"

    path = reporter.save_translated_report(translated_text)

    assert path.exists()
    assert "translated" in path.name
    assert path.suffix == ".txt"
    assert path.read_text(encoding="utf-8") == translated_text


def test_save_enhanced_json_report(tmp_path: Path) -> None:
    """Test saving enhanced JSON report."""
    reporter = MeetingReporter(output_dir=str(tmp_path))
    enhanced_json = {
        "生成日時": datetime.now().isoformat(),
        "投資案件": "テスト案件",
    }

    path = reporter.save_enhanced_json_report(enhanced_json)

    assert path.exists()
    assert "enhanced" in path.name
    assert path.suffix == ".json"

    saved_data = json.loads(path.read_text(encoding="utf-8"))
    assert saved_data["投資案件"] == "テスト案件"
