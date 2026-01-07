"""Report generation for investment meeting results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import google.genai as genai

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
            "投資会議レポート",
            "=" * 80,
            f"日時: {timestamp}",
            f"案件: {case_title}",
            "",
            "参加者:",
            *[f"  - {agent}" for agent in debate_history.agents],
            "",
            "=" * 80,
            "投資案件の概要",
            "=" * 80,
            debate_history.investment_case,
            "",
            "=" * 80,
            "議論の記録",
            "=" * 80,
        ]

        # Add each round
        for round_obj in debate_history.rounds:
            if round_obj.round_num == 0:
                report_parts.append("\n[フェーズ1: 初期意見]\n")
            else:
                report_parts.append(f"\n[フェーズ2: ディベート ラウンド {round_obj.round_num}]\n")

            for msg in round_obj.messages:
                report_parts.append(f">>> {msg.agent_name}")
                report_parts.append(msg.message)
                report_parts.append("")

        # Add consensus
        report_parts.extend(
            [
                "=" * 80,
                "最終合意と リスク評価",
                "=" * 80,
                debate_history.final_consensus or "合意に至りませんでした",
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
            "生成日時": datetime.now().isoformat(),
            "投資案件": debate_history.investment_case,
            "参加者": debate_history.agents,
            "ディベートラウンド": [
                {
                    "ラウンド番号": round_obj.round_num,
                    "メッセージ": [
                        {
                            "エージェント": msg.agent_name,
                            "投資哲学": msg.philosophy,
                            "発言内容": msg.message,
                        }
                        for msg in round_obj.messages
                    ],
                }
                for round_obj in debate_history.rounds
            ],
            "最終合意": debate_history.final_consensus,
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

    async def translate_report_to_japanese(
        self,
        report_content: str,
        model_name: str = "gemini-2.5-flash-lite",
    ) -> str:
        """Translate and enhance report content to natural Japanese.

        Args:
            report_content: Original report text
            model_name: Gemini model to use

        Returns:
            Translated and refined Japanese report
        """
        client = genai.Client()

        prompt = f"""以下の投資会議レポートを、より自然で読みやすい日本語に翻訳・改良してください。
特に以下の点に注意してください:
1. 専門用語や業界用語は適切に日本語化してください
2. 文体は丁寧で分かりやすくしてください
3. レポートの構造と意味は保持してください
4. 数字や重要な情報は正確に保ってください

レポート内容:
{report_content}

翻訳後のレポート:"""

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"max_output_tokens": 2000, "temperature": 0.3},
        )

        if not response.text:
            raise ValueError("翻訳に失敗しました")

        return response.text

    async def enhance_json_report(
        self,
        json_content: str,
        model_name: str = "gemini-2.5-flash-lite",
    ) -> dict[str, Any]:
        """Enhance JSON report with improved Japanese translations.

        Args:
            json_content: Original JSON report
            model_name: Gemini model to use

        Returns:
            Enhanced JSON report as dictionary
        """
        client = genai.Client()

        prompt = f"""以下のJSON形式の投資会議レポートの各フィールドを、より自然で読みやすい日本語に改善してください。
特に以下を強調してください:
1. 各発言の要点を日本語で簡潔にまとめてください
2. 専門用語は分かりやすく説明してください
3. JSON構造は保持してください
4. 元のデータはすべて保持してください

JSON内容:
{json_content}

改善後のJSON (ensure_ascii=false で日本語を含む):"""

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"max_output_tokens": 3000, "temperature": 0.3},
        )

        if not response.text:
            raise ValueError("JSON翻訳に失敗しました")

        # Extract JSON from response
        json_text = response.text
        try:
            result: dict[str, Any] = json.loads(json_text)
            return result
        except json.JSONDecodeError:
            # If response contains markdown code blocks, extract JSON
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()
            result = json.loads(json_text)
            return result

    def save_translated_report(
        self,
        translated_text: str,
        filename_base: str = "investment_meeting",
    ) -> Path:
        """Save translated text report.

        Args:
            translated_text: Translated report content
            filename_base: Base name for output file

        Returns:
            Path to saved translated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_base}_translated_{timestamp}.txt"
        path = self.output_dir / filename
        path.write_text(translated_text, encoding="utf-8")
        return path

    def save_enhanced_json_report(
        self,
        enhanced_json: dict[str, Any],
        filename_base: str = "investment_meeting",
    ) -> Path:
        """Save enhanced JSON report.

        Args:
            enhanced_json: Enhanced JSON report data
            filename_base: Base name for output file

        Returns:
            Path to saved enhanced JSON report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_base}_enhanced_{timestamp}.json"
        path = self.output_dir / filename
        path.write_text(json.dumps(enhanced_json, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def print_summary(
        self,
        debate_history: DebateHistory,
    ) -> None:
        """Print executive summary to console.

        Args:
            debate_history: Complete debate history
        """
        print("\n" + "=" * 80)
        print("投資会議 - サマリー")
        print("=" * 80)
        print(f"案件: {debate_history.investment_case[:100]}...")
        print(f"参加者: {', '.join(debate_history.agents)}")
        print(f"ディベートラウンド: {len(debate_history.rounds) - 1}")  # Exclude Phase 1
        print("\n最終合意:")
        print(debate_history.final_consensus or "未生成")
        print("=" * 80)
