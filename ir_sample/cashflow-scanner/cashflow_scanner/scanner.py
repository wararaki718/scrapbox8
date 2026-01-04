import json
import fitz  # PyMuPDF
import google.generativeai as genai
from pydantic import ValidationError

from .models import FinancialData

class CashFlowScanner:
    """
    Extracts cash flow data from financial reports (text or PDF)
    using Google Gemini API with JSON Mode.
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Google API key is required.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')

    def _get_text_from_pdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return ""

    def extract_financial_data(self, file_path: str) -> FinancialData | None:
        if file_path.lower().endswith('.pdf'):
            content = self._get_text_from_pdf(file_path)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except FileNotFoundError:
                print(f"Error: File not found at {file_path}")
                return None
            except Exception as e:
                print(f"Error reading text file: {e}")
                return None

        if not content:
            print("Error: No content to process.")
            return None

        prompt = f"""
        以下の決算短信テキストから、キャッシュフローに関する数値をJSON形式で抽出してください。
        単位は「百万円」です。数値のみを抽出し、整数で返してください。

        - 営業活動によるキャッシュフロー (operating_cf)
        - 投資活動によるキャッシュフロー (investing_cf)
        - 配当金支払額 (dividend_payout)

        テキスト：
        ---
        {content[:4000]}
        ---

        JSON出力：
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            data = json.loads(response.text)
            return FinancialData(**data)
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            print(f"An error occurred during data extraction: {e}")
            if 'response' in locals():
                 print(f"Raw response: {response.text}")
            return None
