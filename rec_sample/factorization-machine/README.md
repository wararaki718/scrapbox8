# Factorization Machine (MovieLens Small)

`rec_sample/factorization-machine` では、MovieLens `ml-latest-small` を使って
2次の Factorization Machine 回帰を学習するサンプルを実行できます。

## files

- `main.py`: 学習と推薦の実行
- `model.py`: FM 本体（予測・MSE・SGD 学習）
- `utils.py`: MovieLens small のダウンロード/読み込み、one-hot 特徴量作成
- `tests/test_factorization_machine.py`: 振る舞いテスト

## setup

```bash
pip install -r requirements.txt
```

## run

```bash
python main.py
```

初回実行時、以下が自動で作成されます。

- `data/ml-latest-small.zip`
- `data/ml-latest-small/ratings.csv`

## test

```bash
python -m pytest tests/test_factorization_machine.py -q
```
