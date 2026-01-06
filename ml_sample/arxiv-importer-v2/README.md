# arXiv Paper Importer v2

arXiv の論文（PDF）から情報を抽出し、Google Gemini API を活用して高品質な PyTorch モデルコードを自動生成するツールです。

このプロジェクトは、リサーチ業務の効率化と、論文の実装再現を迅速に行うことを目的としています。

## 特徴

- **自動ダウンロード**: arXiv ID または URL から PDF を自動的に取得。
- **構造的テキスト抽出**: `pypdf` を用いて、論文のセクション構造を意識したテキスト抽出を実行。
- **Gemini 2.5 Flash Lite 連携**: 最新の軽量・高速な LLM を用い、論文の数式やアーキテクチャ記述を PyTorch コードに変換。
- **標準化されたワークフロー**: `Makefile` によるセットアップ、静的解析、テストの自動化。
- **高品質なコード出力**: 型ヒントの付与、テンソル形状の追跡コメント、および構文検証済みコードの生成。

## 必要条件

- Python 3.11 以上
- Google Gemini API キー (`GEMINI_API_KEY`)

## セットアップ

1.  **環境変数の設定**:
    Gemini API キーを環境変数としてエクスポートします。
    ```bash
    export GEMINI_API_KEY='your-api-key-here'
    ```

2.  **依存関係のインストール**:
    `Makefile` を使用して、必要なライブラリをインストールします。
    ```bash
    make install
    ```

## 使用方法

### コード生成の実行

以下のコマンドを実行し、プロンプトに従って arXiv の URL を入力してください。

```bash
make run
```

生成されたコードは `generated_models/` ディレクトリに保存されます。

## 開発ワークフロー

品質を維持するために、以下の `Makefile` ターゲットが用意されています。

- `make lint`: `ruff` によるコードスタイルチェックと修正。
- `make typecheck`: `mypy` による厳格な静的型チェック。
- `make test`: `pytest` による生成モデルの順伝播テスト。
- `make check`: 上記すべてのチェックを一括実行。

## プロジェクト構造

```text
arxiv-importer-v2/
├── Makefile                # 標準化されたワークフロー定義
├── pyproject.toml           # ruff, mypy, pytest の設定
├── src/
│   ├── main.py             # エントリポイント
│   ├── extracter.py        # PDF取得・テキスト抽出
│   └── generator.py        # Geminiによるコード生成
├── tests/
│   └── test_model.py       # 生成されたモデルの自動検証テスト
├── downloads/              # ダウンロードされたPDF
└── generated_models/       # 生成されたPyTorchモジュール
```

## エンジニアリング基準

このプロジェクト（v2）では、以下の基準を厳格に適用しています。

1.  **厳格な型ヒント**: すべての関数・メソッドに適切な型ヒントを付与。
2.  **静的解析**: `ruff` を用いたリンティングとフォーマット。
3.  **テンソル形状の追跡**: 生成されるコードには、各演算ごとのテンソル次元をコメントとして記述（LLM への指示に含まれています）。
4.  **自動検証**: 生成されたモデルに対して、ダミーデータを用いた Forward Pass テストの実施。
