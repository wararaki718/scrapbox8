# Simple Specification: README Procedure Generator

Target directory: llm_sample/sample-openai-agent-app

## 1. Purpose
README を入力として受け取り、初心者向けの実行手順書を Markdown で生成する CLI アプリを作る。

## 2. Scope (v1)
含む:
- 単一 README ファイルを入力
- 手順書を生成して標準出力に表示
- 必要に応じてファイル保存

技術固定条件:
- Agent SDK は `openai-agents` を利用する
- モデルは Ollama の `gemma4:e2b` を利用する

含まない:
- Web UI
- コマンドの自動実行
- 複数ファイル横断解析

## 3. CLI Interface
実行コマンド:

```shell
python main.py --input README.md [--output guide.md] [--lang ja|en] [--verbose]
```

引数:
- `--input` (必須): 解析対象 README のパス
- `--output` (任意): 出力先ファイル
- `--lang` (任意): 出力言語。デフォルトは `ja`
- `--verbose` (任意): 処理ログを表示

## 4. Input/Output
入力:
- Markdown テキスト (README)

出力:
- 以下セクションを持つ Markdown
  - Prerequisites
  - Setup
  - Run
  - Verification
  - Troubleshooting

不足情報の扱い:
- 情報が不足している箇所は `Needs confirmation` と明記する。

## 5. Behavior
1. `--input` のファイル存在と読取可否を確認する。
2. README 内容を openai-agents に渡して手順書を生成する。
3. 結果を標準出力へ表示する。
4. `--output` 指定時はファイルにも保存する。

## 6. Error Handling
- ファイルが存在しない場合: エラーメッセージを表示して終了コード 1。
- API 呼び出し失敗時: 原因を表示し、再試行を促して終了コード 1。
- 空の README の場合: 入力不足を表示して終了コード 1。

## 7. Non-Functional
- できるだけシンプルな構成にする。
- 公開関数には型ヒントを付ける。
- 典型的な README サイズで実用的な応答時間を目指す。
- ローカル実行時は Ollama が起動済みで、`gemma4:e2b` が利用可能であることを前提とする。

## 8. Done Criteria
- 指定した README から手順書を生成できる。
- `--output` で保存できる。
- 欠落情報を `Needs confirmation` として明示できる。
- エラー時に理由が分かるメッセージを返せる。
