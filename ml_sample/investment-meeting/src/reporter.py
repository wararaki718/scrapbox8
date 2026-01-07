"""Report generation for investment meeting results."""

import json
from datetime import datetime
from pathlib import Path

from .moderator import DebateHistory


class MeetingReporter:
    """Generate comprehensive reports from investment meeting debates."""

    def __init__(
        self,
        output_dir: str | None = None,
    ) -> None:
        """Initialize reporter.

        Args:
            output_dir: Directory for output reports
        """
        self.output_dir = Path(output_dir or "./reports")
        self.output_dir.mkdir(exist_ok=True)

    def generate_report(
        self,
        debate_history: DebateHistory,
        case_title: str = "Investment Case Analysis",
    ) -> str:
        """Generate comprehensive text report.

        Args:
            debate_history: Complete debate history
            case_title: Title of the investment case

        Returns:
            Formatted report string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report_parts = [
            "=" * 80,
            "INVESTMENT MEETING REPORT",
            "=" * 80,
            f"Date: {timestamp}",
            f"Case: {case_title}",
            "",
            "PARTICIPANTS:",
            *[f"  - {agent}" for agent in debate_history.agents],
            "",
            "=" * 80,
            "INVESTMENT OPPORTUNITY",
            "=" * 80,
            debate_history.investment_case,
            "",
            "=" * 80,
            "DEBATE TRANSCRIPT",
            "=" * 80,
        ]

        # Add each round
        for round_obj in debate_history.rounds:
            if round_obj.round_num == 0:
                report_parts.append("\n[PHASE 1: INITIAL POSITIONS]\n")
            else:
                report_parts.append(
                    f"\n[PHASE 2: DEBATE ROUND {round_obj.round_num}]\n"
                )

            for msg in round_obj.messages:
                report_parts.append(f">>> {msg.agent_name}")
                report_parts.append(msg.message)
                report_parts.append("")

        # Add consensus
        report_parts.extend(
            [
                "=" * 80,
                "FINAL CONSENSUS & RISK ASSESSMENT",
                "=" * 80,
                debate_history.final_consensus or "No consensus reached",
                "",
            ]
        )

        return "\n".join(report_parts)

    def generate_json_report(
        self,
        debate_history: DebateHistory,
    ) -> str:
        """Generate JSON report for programmatic access.

        Args:
            debate_history: Complete debate history

        Returns:
            JSON formatted report
        """
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "investment_case": debate_history.investment_case,
            "participants": debate_history.agents,
            "debate_rounds": [
                {
                    "round_num": round_obj.round_num,
                    "messages": [
                        {
                            "agent": msg.agent_name,
                            "philosophy_snippet": msg.philosophy,
                            "content": msg.message,
                        }
                        for msg in round_obj.messages
                    ],
                }
                for round_obj in debate_history.rounds
            ],
            "final_consensus": debate_history.final_consensus,
        }

        return json.dumps(report_data, indent=2, ensure_ascii=False)

    def save_report(
        self,
        debate_history: DebateHistory,
        filename_base: str = "investment_meeting",
    ) -> tuple[Path, Path]:
        """Save both text and JSON reports to files.

        Args:
            debate_history: Complete debate history
            filename_base: Base name for output files

        Returns:
            Tuple of (text_report_path, json_report_path)
        """
        # Generate reports
        text_report = self.generate_report(debate_history)
        json_report = self.generate_json_report(debate_history)

        # Save text report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_path = self.output_dir / f"{filename_base}_{timestamp}.txt"
        text_path.write_text(text_report, encoding="utf-8")

        # Save JSON report
        json_path = self.output_dir / f"{filename_base}_{timestamp}.json"
        json_path.write_text(json_report, encoding="utf-8")

        return text_path, json_path

    def print_summary(
        self,
        debate_history: DebateHistory,
    ) -> None:
        """Print executive summary to console.

        Args:
            debate_history: Complete debate history
        """
        print("\n" + "=" * 80)
        print("MEETING SUMMARY")
        print("=" * 80)
        print(f"Case: {debate_history.investment_case[:100]}...")
        print(f"Participants: {', '.join(debate_history.agents)}")
        print(f"Debate Rounds: {len(debate_history.rounds) - 1}")  # Exclude Phase 1
        print("\nFinal Consensus:")
        print(debate_history.final_consensus or "Not generated")
        print("=" * 80)
