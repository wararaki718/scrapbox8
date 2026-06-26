# Matrix Factorization (MovieLens)

このディレクトリでは MovieLens データで matrix factorization を実行できます。

- `sample`: 同梱の小規模サンプル (`data/ratings_sample.csv`)
- `official`: MovieLens公式 `ml-latest-small` を自動ダウンロードして利用

## Files

- `data/ratings_sample.csv` : 小規模サンプル評価データ
- `utils.py` : サンプル/公式データの読み込み、公式データの自動ダウンロード
- `model.py` : SGDによる行列分解モデル
- `main.py` : 学習と推薦の実行
- `tests/test_matrix_factorization.py` : 振る舞いテスト

## Setup

```bash
pip install -r requirements.txt
```

## Run (公式データ: 自動ダウンロード)

```bash
python main.py
# または
python main.py --dataset official
```

初回実行時に以下が作成されます。
- `data/ml-latest-small.zip`
- `data/ml-latest-small/ratings.csv`

## Run (同梱サンプル)

```bash
python main.py --dataset sample
```

## Test

```bash
python -m pytest tests/test_matrix_factorization.py -q
```
