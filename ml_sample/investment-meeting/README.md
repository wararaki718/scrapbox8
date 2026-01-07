# Investment Meeting Simulator

投資判断「ベテラン会議」シミュレーター。Google Gemini 2.5 Flash Lite を活用し、異なる投資哲学を持つ複数のAIエージェントが議論を行い、投資判断のバイアスを排除するための意思決定支援ツール。

## 🎯 プロジェクト概要

3人の異なる投資家エージェントが、**最低3往復の徹底的な議論**を通じて投資判断を行うシステムです。

### エージェント

1. **グロース投資家 (Growth Investor)**
   - 市場の破壊的イノベーションに焦点
   - 将来のキャッシュフロー、市場シェア拡大を重視
   - 高い成長率、TAM拡大を評価

2. **バリュー投資家 (Value Investor)**
   - 安全域 (Margin of Safety) を重視
   - PBR/PER などの財務指標を重視
   - B/S の健全性、キャッシュフロー安定性を評価

3. **データ重視アナリスト (Data Analyst)**
   - マクロ指標と直近決算データに基づく
   - 競合比較、業界ベンチマーク
   - 中立・批判的な分析姿勢

## 📋 ディレクトリ構造

```
investment-meeting/
├── README.md
├── Makefile                 # make install, make check, make run
├── pyproject.toml          # プロジェクト設定 (ruff, mypy)
├── requirements.txt        # 依存パッケージ
├── src/
│   ├── __init__.py
│   ├── agents.py           # 投資家エージェント定義
│   ├── moderator.py        # 議論ロジック（3往復強制）
│   ├── reporter.py         # 議論結果の集約・レポート生成
│   └── main.py             # エントリーポイント
└── tests/
    ├── __init__.py
    ├── test_agents.py      # エージェント動作テスト
    ├── test_moderator.py   # 3往復ループの検証
    └── test_integration.py # 統合テスト
```

## 🔄 議論の進行フロー

### フェーズ1: 意見提示 (Phase 1: Initial Opinions)
各エージェントが投資機会に対する初期判断を述べます。
- **並列実行**: asyncio で3つのエージェントが同時に意見を生成
- **出力**: 各投資家の初期スタンス

### フェーズ2: 3往復の対立化ディベート (Phase 2: Multi-Round Debate)

**最低3ラウンド**の強制ループを実行します：

#### ラウンド 1
- 各エージェントが前の発言者の論理的弱点を指摘
- 例）グロース投資家がバリュー投資家の「過度な慎重さ」を批判

#### ラウンド 2
- バリュー投資家がグロース投資家の「リスク軽視」を指摘
- アナリストが双方の仮定の妥当性を検証

#### ラウンド 3
- 最終的な立場の表明と、譲歩できない点の明確化
- 対立点の整理

**安易な同意禁止**：プロンプトで以下を強制
```
1. You MUST identify at least 1 logical weakness or blind spot
2. You MUST articulate why your investment philosophy is more appropriate
3. Do NOT simply agree with others
4. Be specific: cite concrete concerns or opportunities
```

### フェーズ3: 合意形成と最終評価 (Phase 3: Consensus Formation)
- 3往復後、データ分析者が最終評価をまとめる
- 上昇リスク、下降リスク、必要データの整理
- **投資判断**: BUY / HOLD / SELL の決定

## 🚀 セットアップと実行

### インストール

```bash
cd ml_sample/investment-meeting

# 依存パッケージをインストール
make install

# または pip で直接インストール
pip install -r requirements.txt
```

### 環境変数の設定

```bash
# export コマンドで環境変数を設定
export GEMINI_API_KEY="your_gemini_api_key_here"

# または、実行時に直接指定
GEMINI_API_KEY="your_gemini_api_key_here" make run
```

[Gemini API キー取得](https://aistudio.google.com/app/apikey)

### 実行

```bash
# 環境変数を設定した上で実行
export GEMINI_API_KEY="your_api_key"
make run

# 直接実行
export GEMINI_API_KEY="your_api_key"
python -m src.main
```

**出力**：
- コンソールに議論の進行状況を表示
- `reports/` ディレクトリに `.txt` と `.json` レポートを保存

## ✅ 品質管理とテスト

### Linting & Formatting

```bash
# ruff でコードをフォーマット
make format

# ruff でコードをチェック
make lint
```

### 型チェック

```bash
# mypy で型チェック
make typecheck
```

### テスト実行

```bash
# pytest を実行（カバレッジ付き）
make test

# 全チェック（ruff, mypy, pytest）
make check
```

**重要なテスト**：
- `test_moderator.py::TestThreeRoundEnforcement`: 3往復ループの検証
- `test_integration.py::TestThreeRoundIntegration`: 全フェーズの完全性確認

## 🌐 AI による日本語翻訳機能

生成されたレポートは、Google Gemini を使用して自動的に日本語に翻訳・改良されます。

### 翻訳プロセス

1. **テキストレポート翻訳** (`translate_report_to_japanese`)
   - 元のレポートをより自然で読みやすい日本語に翻訳
   - 専門用語の適切な日本語化
   - 元の情報構造を保持しながら、可読性を向上

2. **JSON レポート強化** (`enhance_json_report`)
   - JSON の各フィールドを改善
   - 複雑な発言内容を簡潔にまとめた日本語要約を追加
   - プログラマティックアクセス可能なまま日本語対応

### 生成されるファイル

```bash
reports/
├── investment_decision_YYYYMMDD_HHMMSS.txt        # 元のテキストレポート
├── investment_decision_YYYYMMDD_HHMMSS.json       # 元の JSON レポート
├── investment_decision_translated_YYYYMMDD_HHMMSS.txt  # 翻訳済みテキスト
└── investment_decision_enhanced_YYYYMMDD_HHMMSS.json   # 改良済み JSON
```

### 使用例

```python
from src.reporter import MeetingReporter

reporter = MeetingReporter()

# 基本レポート生成
text_path, json_path = reporter.save_report(debate_history)

# テキストレポートを日本語に翻訳
text_content = text_path.read_text(encoding="utf-8")
translated = await reporter.translate_report_to_japanese(text_content)
translated_path = reporter.save_translated_report(translated)

# JSON レポートを改良
json_content = json_path.read_text(encoding="utf-8")
enhanced = await reporter.enhance_json_report(json_content)
enhanced_path = reporter.save_enhanced_json_report(enhanced)
```

### 翻訳品質設定

翻訳は以下の設定で行われます：
- **モデル**: `gemini-2.5-flash-lite`
- **温度**: 0.3（確定性重視、創造性を低く）
- **最大トークン**: 2000（テキスト）、3000（JSON）

## 📊 レポート出力例

### テキストレポート (`investment_decision_YYYYMMDD_HHMMSS.txt`)

```
================================================================================
投資会議レポート
================================================================================
日時: 2026-01-07 14:30:45
案件: TechCloud Inc.

参加者:
  - Growth Investor
  - Value Investor
  - Data Analyst

================================================================================
投資案件の概要
================================================================================
Evaluate investment opportunity: TechCloud Inc.
...

================================================================================
議論の記録
================================================================================

[フェーズ1: 初期意見]

>>> Growth Investor
This is a compelling opportunity to capture emerging market share...
[続く]
```

### JSON レポート (`investment_decision_YYYYMMDD_HHMMSS.json`)

```json
{
  "生成日時": "2026-01-07T14:30:45",
  "投資案件": "...",
  "参加者": ["Growth Investor", "Value Investor", "Data Analyst"],
  "ディベートラウンド": [
    {
      "ラウンド番号": 0,
      "メッセージ": [
        {
          "エージェント": "Growth Investor",
          "投資哲学": "...",
          "発言内容": "..."
        }
      ]
    }
  ],
  "最終合意": "..."
}
```

### 翻訳済みレポート (`investment_decision_translated_YYYYMMDD_HHMMSS.txt`)

自動生成されたテキストレポートを、より自然で読みやすい日本語に翻訳・改善したバージョンです。

### 強化済み JSON (`investment_decision_enhanced_YYYYMMDD_HHMMSS.json`)

各フィールドの日本語翻訳を改善し、より理解しやすくした JSON 形式のレポートです。

## 🔧 カスタマイズ

### 投資ケースの変更

`main.py` の `investment_case` 変数を編集：

```python
investment_case = """
Evaluate investment in [Company Name]:
- [Financial metrics]
- [Competitive landscape]
- [Key risks]
"""
```

### エージェント数の拡張

`agents.py` に新しいエージェントクラスを追加：

```python
class MacroeconomistAgent(InvestmentAgent):
    def __init__(self):
        philosophy = """
        You are a macroeconomist who...
        """
        super().__init__(name="Macro Economist", philosophy=philosophy)
```

`main.py` でインスタンス化：

```python
agents = [
    GrowthInvestor(),
    ValueInvestor(),
    DataAnalyst(),
    MacroeconomistAgent(),  # 新規追加
]
```

### ディベートラウンド数の変更

`moderator.py` で最小ラウンド数を変更：

```python
class InvestmentMeetingModerator:
    MIN_DEBATE_ROUNDS = 5  # 3 から 5 に変更
```

## 🏗️ アーキテクチャ設計

### async/await パターン

エージェントへのリクエストは非同期で並列実行されます：

```python
# Phase 1: 3つのエージェントが同時に生成
tasks = [
    agent.generate_response(prompt) 
    for agent in agents
]
messages = await asyncio.gather(*tasks)
```

### Pydantic モデル

すべてのデータ構造は型安全に定義：

```python
class AgentMessage(BaseModel):
    agent_name: str
    philosophy: str
    message: str
    round_num: int

class DebateRound(BaseModel):
    round_num: int
    messages: list[AgentMessage]

class DebateHistory(BaseModel):
    investment_case: str
    agents: list[str]
    rounds: list[DebateRound]
    final_consensus: Optional[str] = None
```

### Gemini API 統合

`google-generativeai` ライブラリで Gemini 2.5 Flash Lite を使用：

```python
model = genai.GenerativeModel("gemini-2.5-flash-lite")
response = model.generate_content(
    prompt,
    generation_config=genai.types.GenerationConfig(
        max_output_tokens=500,
        temperature=0.7,
    ),
)
```

## 📈 パフォーマンス考慮事項

- **並列化**: 各フェーズで非同期リクエスト
- **トークン制限**: 各レスポンスは最大 500 トークン
- **キャッシュ戦略**: 会話履歴はメモリに保持

## 🐛 トラブルシューティング

### API キーエラー

```
ValueError: GOOGLE_API_KEY environment variable not set
```

**対応**: `.env` ファイルを作成し、`GOOGLE_API_KEY` を設定してください。

### 型チェックエラー

```bash
make typecheck
```

すべての関数に型ヒントが付与されていることを確認。

### テスト失敗

```bash
make test
```

特に `test_moderator.py::TestThreeRoundEnforcement` で 3 ラウンドが強制されていることを確認。

## 📚 参考資料

- [Google Gemini API Documentation](https://ai.google.dev/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [asyncio Documentation](https://docs.python.org/3/library/asyncio.html)

## 📝 ライセンス

MIT License

## 👨‍💼 作者

Investment AI Lab

---

**最後の確認**: 本プロジェクトは最低 3 往復の議論ループを強制し、エージェント間の対立を最大化するように設計されています。安易な同意は避けられ、各エージェントは自身の投資哲学を固持して議論を戦わせます。
