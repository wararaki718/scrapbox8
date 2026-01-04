import os
from pathlib import Path

from cashflow_scanner.downloader import PDFDownloader
from cashflow_scanner.scanner import CashFlowScanner
from cashflow_scanner.analysis import calculate_dividend_safety_score

def main():
    """
    Main function to run the cash flow scanner application.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    download_dir = Path("downloads")
    
    pdf_url = input("Enter the URL of the financial report PDF: ")
    if not pdf_url:
        print("No URL provided. Exiting.")
        return

    downloader = PDFDownloader()
    downloaded_pdf_path = downloader.download(pdf_url, download_dir)

    if not downloaded_pdf_path:
        return

    try:
        scanner = CashFlowScanner(api_key=api_key)
        financial_data = scanner.extract_financial_data(str(downloaded_pdf_path))

        if financial_data:
            print("\n--- 抽出された財務データ ---")
            print(f"営業CF: {financial_data.operating_cf} 百万円")
            print(f"投資CF: {financial_data.investing_cf} 百万円")
            print(f"配当金支払額: {financial_data.dividend_payout} 百万円")
            print("-" * 25)

            score, judgment = calculate_dividend_safety_score(financial_data)
            
            free_cf = financial_data.operating_cf + financial_data.investing_cf
            print(f"フリーキャッシュフロー: {free_cf} 百万円")
            print(f"配当安全性スコア: {score:.2f}")
            print(f"判定: {judgment}")
            print("-" * 25)
    finally:
        if downloaded_pdf_path and os.path.exists(downloaded_pdf_path):
            os.remove(downloaded_pdf_path)
            print(f"Cleaned up {downloaded_pdf_path}")

if __name__ == "__main__":
    main()

