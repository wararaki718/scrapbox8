"""論文テキストからPyTorchモデルコードを生成します."""

import ast
import logging
from typing import Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)


class CodeGenerator:
    """抽出された論文テキストから高品質なPyTorchモデルコードを生成するクラス."""

    def __init__(self, api_key: str) -> None:
        """CodeGeneratorを初期化します.

        Args:
            api_key: Google APIキー.

        """
        if not api_key:
            raise ValueError("Google API key is required.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")

    def generate_pytorch_module(
        self, paper_text: str, model_name: str
    ) -> Optional[str]:
        """論文のテキストからPyTorchのnn.Moduleを生成します."""
        system_prompt = """
        あなたは、世界トップレベルのAI研究所に所属するシニアリサーチエンジニアです。
        あなたの任務は、論文の抽象的な数式、図、アーキテクチャ記述を、再現性と保守性の高いPyTorchコードに変換することです。
        単なるコード生成ではなく、論文の核心的なアイデアを忠実に実装するアーキテクトとして振る舞ってください。

        **遵守すべき実装原則:**
        1.  **クラス定義**: `torch.nn.Module` を継承したクラスを生成してください。クラス名は `{model_name}` とします。
        2.  **初期化 (`__init__`)**:
            -   論文で提案されているハイパーパラメータ（次元数、ヘッド数、レイヤー数など）を引数として受け取ってください。
            -   `torch.nn.Linear`, `torch.nn.Conv2d`, `torch.nn.LayerNorm` などの層を定義してください。
        3.  **フォワードパス (`forward`)**:
            -   データの流れる順序を正確に実装してください。
            -   **Shape Tracking**: 各演算の直後に、その時点でのテンソルの形状をコメントで記述してください。例: `# x: (B, Seq, Dim)`
            -   論文中の数式番号やセクション名をコメントとして付与し、コードと論文の対応関係を明確にしてください。例: `# Eq. (1) in Section 3.1`
        4.  **PyTorchベストプラクティス**:
            -   デバイス指定に柔軟に対応できるよう、`forward` メソッド内で新たに生成するテンソルには `.to(device)` を適用してください。
            -   数値的安定性を考慮し、分類の最終層では `F.log_softmax` の利用を検討してください。
        5.  **出力形式**:
            -   最終的な出力は、必要なimport文を含む完全なPythonコードブロックのみとしてください。
            -   コードの前に説明や言い訳は不要です。
        """

        prompt = f"""
        以下の論文テキストから、モデル名 `{model_name}` としてPyTorchの`nn.Module`を実装してください。

        --- 論文テキスト ---
        {paper_text[:32000]}
        ---
        """

        try:
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-latest", generation_config=generation_config
            )
            response = model.generate_content(prompt)
            # response.textがNoneの場合も考慮
            generated_code = str(response.text) if response.text else None
        except Exception as e:
            logger.error("Error during code generation: %s", e)
            return None

        if not generated_code:
            logger.warning("Generated code is empty.")
            return None

    def validate_syntax(self, code: str) -> bool:
        """生成されたPythonコードの構文が正しいかチェックします."""
        if not code:
            return False
        try:
            ast.parse(code)
            logger.info("Generated code syntax is valid.")
            return True
        except SyntaxError as e:
            logger.error("Generated code has a syntax error: %s", e)
            return False
