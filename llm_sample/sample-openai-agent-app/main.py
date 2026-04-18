from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn

from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from openai import AsyncOpenAI

DEFAULT_MODEL = "gemma4:e2b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a beginner-friendly run guide from README using openai-agents + Ollama."
    )
    parser.add_argument("--input", required=True, help="Path to input README markdown file")
    parser.add_argument("--output", help="Optional output markdown file path")
    parser.add_argument("--lang", choices=["ja", "en"], default="ja", help="Output language")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs")
    return parser.parse_args()


def vlog(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[verbose] {message}", file=sys.stderr)


def fail(message: str) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_readme(path: Path) -> str:
    if not path.exists() or not path.is_file():
        fail(f"Input file not found: {path}")

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"Failed to read input file: {exc}")

    if not content.strip():
        fail("Input README is empty.")

    return content


def build_prompt(readme_text: str, lang: str) -> str:
    language_instruction = (
        "Write all explanatory text in Japanese." if lang == "ja" else "Write all text in English."
    )

    return f"""
You are a technical documentation assistant.
Generate a beginner-friendly setup/run guide from the README below.

Hard requirements:
- Keep exactly these markdown headings in this order:
  1. Prerequisites
  2. Setup
  3. Run
  4. Verification
  5. Troubleshooting
- If information is missing, explicitly write: Needs confirmation
- Do not invent environment-specific values.
- Normalize command order when README order is unclear.
- Include concrete shell commands when available in the README.
- Keep output concise and practical.
- {language_instruction}

README:
---
{readme_text}
---
""".strip()


def create_agent() -> Agent:
    base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    api_key = os.environ.get("OLLAMA_API_KEY", "ollama")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    model = OpenAIChatCompletionsModel(model=DEFAULT_MODEL, openai_client=client)

    return Agent(
        name="readme-procedure-generator",
        instructions="Create reliable, beginner-friendly runbooks from README input.",
        model=model,
    )


def run_generation(readme_text: str, lang: str, verbose: bool) -> str:
    vlog(verbose, "Disabling tracing to avoid external tracing dependencies")
    set_tracing_disabled(True)

    agent = create_agent()
    prompt = build_prompt(readme_text, lang)

    vlog(verbose, f"Calling agent model={DEFAULT_MODEL}")
    try:
        result = Runner.run_sync(agent, prompt)
    except Exception as exc:  # noqa: BLE001
        fail(
            "Agent call failed. Verify Ollama is running and model gemma4:e2b is available. "
            f"Details: {exc}"
        )

    output = result.final_output if hasattr(result, "final_output") else str(result)
    if not isinstance(output, str):
        output = str(output)

    if not output.strip():
        fail("Agent returned empty output.")

    return output


def write_output(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        fail(f"Failed to write output file: {exc}")


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    readme_text = read_readme(input_path)
    guide = run_generation(readme_text=readme_text, lang=args.lang, verbose=args.verbose)

    print(guide)

    if args.output:
        output_path = Path(args.output)
        vlog(args.verbose, f"Writing output file: {output_path}")
        write_output(output_path, guide)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
