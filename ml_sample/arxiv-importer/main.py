import os
from pathlib import Path
from urllib.parse import urlparse
from modules.extracter import PaperExtractor
from modules.generator import CodeGenerator

def get_arxiv_id_from_url(url: str) -> str:
    """
    arXivのURLから論文IDを抽出する。
    例: "https://arxiv.org/abs/1706.03762" -> "1706.03762"
    """
    parsed_url = urlparse(url)
    return parsed_url.path.split('/')[-1]

def main():
    """
    メイン処理：arXiv論文からPyTorchコードを生成する。
    """
    # 1. APIキーの設定
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    # 2. ユーザーからarXivのURLを入力
    arxiv_url = input("Enter the arXiv paper URL (e.g., https://arxiv.org/abs/1706.03762): ")
    if not arxiv_url:
        print("No URL provided. Exiting.")
        return

    arxiv_id = get_arxiv_id_from_url(arxiv_url)
    
    # 3. 論文のダウンロードとテキスト抽出
    download_dir = Path("downloads")
    output_dir = Path("generated_models")
    output_dir.mkdir(exist_ok=True)

    extractor = PaperExtractor(download_dir=download_dir)
    pdf_path = extractor.download_from_arxiv(arxiv_id)

    if not pdf_path:
        return

    paper_text = extractor.extract_text_from_pdf(pdf_path)
    if not paper_text:
        print("Failed to extract text from the paper.")
        return

    # 4. PyTorchコードの生成
    generator = CodeGenerator(api_key=api_key)
    model_name = f"Model_{arxiv_id.replace('.', '_')}"
    generated_code = generator.generate_pytorch_module(paper_text, model_name=model_name)

    if not generated_code:
        print("Code generation failed.")
        return

    # 5. 生成コードの構文チェックと保存
    if generator.validate_syntax(generated_code):
        output_path = output_dir / f"{model_name}.py"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(generated_code)
        print(f"\nSuccessfully generated and saved the model to: {output_path}")
        print("\n--- Generated Code ---")
        print(generated_code)
    else:
        print("\n--- Invalid Generated Code ---")
        print(generated_code)


if __name__ == "__main__":
    main()
