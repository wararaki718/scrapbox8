"""arXiv論文からPyTorchコードを生成するメイン処理."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# `src`ディレクトリをシステムパスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.extracter import PaperExtractor  # noqa: E402
from src.generator import CodeGenerator  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_arxiv_id_from_url(url: str) -> str:
    """arXivのURLから論文IDを抽出します.

    例: "https://arxiv.org/abs/1706.03762" -> "1706.03762"
    """
    return url.split("/")[-1]


def save_code(code: str, output_dir: Path, model_name: str) -> Path:
    """生成されたコードを指定されたディレクトリに保存します."""
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{model_name}.py"
    with output_path.open("w", encoding="utf-8") as f:
        f.write(code)
    return output_path


def main() -> None:
    """メイン処理：arXiv論文からPyTorchコードを生成します."""
    api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Error: GEMINI_API_KEY environment variable not set.")
        return

    arxiv_url: str = input(
        "Enter the arXiv paper URL (e.g., https://arxiv.org/abs/1706.03762): "
    )
    if not arxiv_url:
        logger.warning("No URL provided. Exiting.")
        return

    arxiv_id = get_arxiv_id_from_url(arxiv_url)

    download_dir = Path("downloads")
    output_dir = Path("generated_models")

    extractor = PaperExtractor(download_dir=download_dir)
    pdf_path = extractor.download_from_arxiv(arxiv_id)

    if not pdf_path:
        return

    paper_text = extractor.extract_structured_text(pdf_path)
    if not paper_text:
        logger.error("Failed to extract text from the paper.")
        return

    generator = CodeGenerator(api_key=api_key)
    model_name = f"Model_{arxiv_id.replace('.', '_')}"
    generated_code = generator.generate_pytorch_module(
        paper_text, model_name=model_name
    )

    if not generated_code:
        logger.error("Code generation failed.")
        return

    if generator.validate_syntax(generated_code):
        output_path = save_code(generated_code, output_dir, model_name)
        logger.info("\nSuccessfully generated and saved the model to: %s", output_path)
        logger.info("\n--- Generated Code ---\n%s", generated_code)
    else:
        logger.error("\n--- Invalid Generated Code ---\n%s", generated_code)


if __name__ == "__main__":
    main()
