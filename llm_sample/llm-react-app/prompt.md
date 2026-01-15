# 指示: Gemini 2.5 Flash-Lite を用いた LangGraph ReAct エージェントの実装

以下のディレクトリ構造に基づき、`gemini-2.5-flash-lite` を活用した自律型エージェントを実装してください。環境変数はシステムの `.zprofile` で定義済みの `GEMINI_API_KEY` を使用します。

## 1. プロジェクト構造
root: `scrapbox8/llm_sample/llm-react-app`
- `main.py`: エントリーポイント
- `src/models.py`: LLM（Gemini）の初期化
- `src/agent.py`: LangGraph のグラフ（ReAct）定義
- `src/tools/github_tools.py`: カスタムツールの定義

## 2. 各ファイルの実装詳細

### A. `src/models.py`
- `langchain-google-genai` から `ChatGoogleGenerativeAI` をインポート。
- モデル名に `gemini-2.5-flash-lite` を指定。
- 環境変数 `GEMINI_API_KEY` を取得し、`google_api_key` 引数に渡すロジックを実装してください。
- `temperature=0` に設定。

### B. `src/tools/github_tools.py`
- `langchain_core.tools` の `@tool` デコレータを使用。
- `get_repository_info(repo_name: str)`: 指定されたリポジトリの概要を返すモック関数。
- `analyze_code_structure(path: str)`: 指定パスのコード構造を分析するモック関数。

### C. `src/agent.py`
- `langgraph.prebuilt.create_react_agent` を使用。
- `src/models.py` のモデルと `src/tools/github_tools.py` のツールを結合。
- `langgraph.checkpoint.memory.MemorySaver` を使用してチェックポインタを有効化。
- システムプロンプトに「あなたは Chain of Thought を用いて GitHub 操作を行うシニアエンジニアです」と設定。

### D. `main.py`
- `src/agent.py` のエージェントを呼び出し。
- 以下の入力で実行し、思考プロセスと最終回答を出力するコードを記述：
  「リポジトリ scrapbox8 の構造を分析し、README の改善案を出してください。」

## 3. 制約事項
- `.env` ファイルや `python-dotenv` は使用せず、`os.environ.get("GEMINI_API_KEY")` を直接参照してください。
- 最新の LangGraph (0.2+) および LangChain (0.3+) のシンタックスを使用してください。
- 各ファイルの内容を個別のコードブロックで出力してください。
