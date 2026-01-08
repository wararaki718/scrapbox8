from pathlib import Path

import requests
from pypdf import PdfReader


class PaperExtractor:
    """arXivから論文をダウンロードし、テキストコンテンツを抽出するクラス."""

    def __init__(self, download_dir: Path = Path("downloads")) -> None:
        """Initialize PaperExtractor.

        Args:
            download_dir: PDFを保存するディレクトリ.

        """
        self.download_dir = download_dir
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download_from_arxiv(self, arxiv_id: str) -> Path | None:
        """arXiv IDからPDFをダウンロードします.

        URLは https://arxiv.org/pdf/{arxiv_id}.pdf の形式。
        """
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        save_path = self.download_dir / f"{arxiv_id}.pdf"

        if save_path.exists():
            print(f"PDF already exists: {save_path}")
            return save_path

        try:
            print(f"Downloading PDF from: {pdf_url}")
            response = requests.get(
                pdf_url, headers=self.headers, stream=True, timeout=60
            )
            response.raise_for_status()

            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Successfully downloaded to: {save_path}")
            return save_path
        except requests.exceptions.RequestException as e:
            print(f"Error downloading PDF: {e}")
            return None

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """PDFファイルからテキストを抽出します.

        数式や特殊な記号の扱いは限定的であることに注意。
        """
        if not pdf_path.exists():
            print(f"Error: PDF file not found at {pdf_path}")
            return ""

        try:
            print(f"Extracting text from {pdf_path}...")
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            # 論文の主要セクションに絞ることで精度向上を狙う (簡易的な実装)
            method_start = text.lower().find("method")
            if method_start != -1:
                text = text[method_start:]

            print("Text extraction complete.")
            return text
        except Exception as e:
            print(f"An error occurred during text extraction: {e}")
            return ""
