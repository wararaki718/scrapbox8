# Gemini 2.5 Flash-Lite ReAct Agent with LangGraph

このサンプルは、Gemini 2.5 Flash-Lite を使用して LangGraph の ReAct エージェントを実装したプロジェクトです。
GitHub のリポジトリ情報の取得や構造の分析を行う自律型エージェントの基本構成を示します。

## 概要

- **LLM**: Gemini 2.5 Flash-Lite (`gemini-2.5-flash-lite`)
- **Framework**: LangGraph (ReAct agent), LangChain
- **Features**:
  - `ChatGoogleGenerativeAI` を用いた高速な推論。
  - `langgraph.prebuilt.create_react_agent` によるグラフ定義。
  - `MemorySaver` によるチェックポイント機能（対話履歴の保持）。
  - カスタムツールによる GitHub 操作のシミュレーション。

## セットアップ

### 1. 環境変数の設定

Google AI Studio で取得した API キーを環境変数 `GEMINI_API_KEY` に設定してください。

```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

## 構成

- `src/main.py`: エントリーポイント。エージェントへの問い合わせと実行結果の出力を行います。
- `src/models.py`: Gemini モデルの初期化設定。
- `src/agent.py`: LangGraph の ReAct グラフ定義。
- `src/tools/github_tools.py`: エージェントが使用するカスタムツール（モック）。

## 実行方法

以下のコマンドを実行すると、リポジトリの構造分析と改善案の生成が開始されます。

```bash
python src/main.py
```
