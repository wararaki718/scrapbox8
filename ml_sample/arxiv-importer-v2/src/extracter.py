"""arXivからの論文抽出機能を提供します."""

import logging
import re
from pathlib import Path
from typing import Optional

import requests
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperExtractor:
    """arXivから論文をダウンロードし、テキストコンテンツを構造的に抽出するクラス."""

    def __init__(self, download_dir: Path = Path("downloads")) -> None:
        """PaperExtractorを初期化します.

        Args:
            download_dir: PDFを保存するディレクトリ.

        """
        self.download_dir = download_dir
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download_from_arxiv(self, arxiv_id: str) -> Optional[Path]:
        """ARXiv IDからPDFをダウンロードします."""
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        save_path = self.download_dir / f"{arxiv_id}.pdf"

        if save_path.exists():
            logger.info("PDF already exists: %s", save_path)
            return save_path

        try:
            logger.info("Downloading PDF from: %s", pdf_url)
            response = requests.get(
                pdf_url, headers=self.headers, stream=True, timeout=60
            )
            response.raise_for_status()

            with save_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info("Successfully downloaded to: %s", save_path)
            return save_path
        except requests.exceptions.RequestException as e:
            logger.error("Error downloading PDF: %s", e)
            return None

    def extract_structured_text(self, pdf_path: Path) -> str:
        """PDFからテキストを抽出し、関連セクションを取得します."""
        if not pdf_path.exists():
            logger.error("Error: PDF file not found at %s", pdf_path)
            return ""

        try:
            logger.info("Extracting structured text from %s...", pdf_path)
            reader = PdfReader(pdf_path)
            full_text = "".join(page.extract_text() or "" for page in reader.pages)

            # 正規表現で主要セクションを特定
            match = re.search(r"(?i)(?:3|4)\s*(?:Method|Architecture|Model)", full_text)
            if match:
                start_index = match.start()
                end_match = re.search(
                    r"(?i)(?:4|5)\s*(?:Experiments|Results)", full_text[start_index:]
                )
                if end_match:
                    relevant_text = full_text[
                        start_index : start_index + end_match.start()
                    ]
                    logger.info(
                        "Found relevant sections. Focusing on Method/Architecture."
                    )
                    return relevant_text

            logger.info("Could not find specific sections, using full text.")
            return full_text
        except Exception as e:
            logger.error("An error occurred during text extraction: %s", e)
            return ""
