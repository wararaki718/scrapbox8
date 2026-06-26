# Learning k-Determinantal Point Processes for Personalized Ranking

論文 "Learning k-Determinantal Point Processes for Personalized Ranking" の
中核アルゴリズムを、最小構成で再現したサンプルです。

この実装では、ユーザーごとの品質項と共有多様性項を使った
Personalized k-DPP を学習し、学習後に多様性を保った推薦を行います。

## Files

- `model.py`: Personalized k-DPP モデル (`L_u = Q_u S Q_u`) の定義
- `loss.py`: `torch.nn` 互換の損失クラス `PersonalizedKDPPNLLLoss`
- `data.py`: k-DPP ベースの合成学習データ生成
- `train.py`: 学習ループ (`train_model`)
- `recommend.py`: 推薦結果表示の補助関数
- `utils.py`: k-DPP の正規化項 $Z_k$（固有値の elementary symmetric polynomial）と対数確率
- `main.py`: 各モジュールを組み合わせる実行エントリ
- `tests/test_lkp_ranking.py`: 主要ロジックのユニットテスト

## Setup

```bash
cd rec_sample/lkp-ranking
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

実行すると以下が確認できます。

- 学習前後の NLL の変化
- 学習ログ（エポックごとの損失）
- ユーザーごとの既知アイテムを除外した推薦結果

## Test

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Notes

- k-DPP の確率は
	$P(Y \mid u, |Y|=k) = \det(L_{u,Y}) / Z_k(L_u)$
	をそのまま使っています。
- 正規化項 $Z_k$ は固有値の $k$ 次 elementary symmetric polynomial で計算しています。
- デモの学習データは、隠れた Personalized k-DPP からサンプルした合成データです。
