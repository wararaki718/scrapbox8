"""Generates PyTorch code from paper text using Google Gemini API."""

import ast
from typing import Optional

import google.genai as genai


class CodeGenerator:
    """抽出された論文テキストからPyTorchモデルコードを生成するクラス."""

    def __init__(self, api_key: str) -> None:
        """Initialize CodeGenerator.

        Args:
            api_key: Google APIキー.

        """
        if not api_key:
            raise ValueError("Google API key is required.")
        self.client = genai.Client(api_key=api_key)

    def generate_pytorch_module(
        self, paper_text: str, model_name: str = "MyModel"
    ) -> Optional[str]:
        """論文のテキストからPyTorchのnn.Moduleを生成します."""
        system_prompt = """
        あなたは深層学習の論文を即座にコードに変換するシニアリサーチエンジニアです。
        論文中の数式、図、文章で説明されているモデルアーキテクチャを
        PyTorchの演算に正確にマッピングし、
        再利用可能なクリーンな`nn.Module`クラスを生成してください。

        満たすべき要件：
        1.  `torch.nn.Module`を継承したクラス定義を生成してください。
        2.  `__init__`メソッドで、論文に記載されている層（Layers）
            を定義してください。
        3.  `forward`メソッドで、データの流れを実装してください。
        4.  生成コードの各ブロックには、論文のどの部分（セクション名、
            数式番号、図番号など）に基づいているかを示す
            コメントを日本語で追記してください。
        5.  `forward`メソッドのDocstringに、テンソルの形状（shape）
            がどのように変化するかを記述してください。
            例：(batch_size, channels, height, width)
            -> (batch_size, new_channels, ...)。
        6.  最終的な出力は、Pythonコードブロックのみとしてください。
            追加の説明は不要です。
        """

        prompt = f"""
        以下の論文テキストから、モデル名 `{model_name}` として
        PyTorchの`nn.Module`を実装してください。

        --- 論文テキスト ---
        {paper_text[:30000]}
        ---
        """

        try:
            print("Generating PyTorch code from paper text...")
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=system_prompt + "\n" + prompt,
            )

            # ```python ... ``` の中身を抽出
            res_text = response.text
            if res_text is None:
                print("Error: Model response text is None.")
                return None

            code = res_text.strip()
            if code.startswith("```python"):
                code = code[9:]
            if code.endswith("```"):
                code = code[:-3]

            print("Code generation complete.")
            return code.strip()
        except Exception as e:
            print(f"An error occurred during code generation: {e}")
            return None

    def validate_syntax(self, code: str) -> bool:
        """生成されたPythonコードの構文が正しいかチェックします."""
        if not code:
            return False
        try:
            ast.parse(code)
            print("Generated code syntax is valid.")
            return True
        except SyntaxError as e:
            print(f"Generated code has a syntax error: {e}")
            return False
