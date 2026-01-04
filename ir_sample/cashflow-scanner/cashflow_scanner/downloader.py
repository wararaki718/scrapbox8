from pathlib import Path
from datetime import datetime
import requests

class PDFDownloader:
    """
    指定されたURLからPDFファイルをダウンロードするクラス
    """
    def __init__(self):
        """
        リクエストヘッダーを初期化
        """
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def download(self, url: str, save_dir: Path) -> Path | None:
        """
        URLからPDFをダウンロードし、指定されたディレクトリに保存する
        """
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.pdf"
            save_path = save_dir / filename

            print(f"Downloading PDF from: {url}")
            response = requests.get(url, headers=self.headers, stream=True, timeout=30)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Successfully downloaded to: {save_path}")
            return save_path

        except requests.exceptions.RequestException as e:
            print(f"Error downloading PDF: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during download: {e}")
            return None
