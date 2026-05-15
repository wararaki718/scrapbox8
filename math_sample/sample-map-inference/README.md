# map inference

ガウス事前付き線形回帰の MAP 推定（Ridge 相当）を、
データ生成と推論アルゴリズムを分離して実装したサンプルです。

## files

- `main.py`: エントリーポイント
- `data_generator.py`: 合成データ生成
- `inference.py`: MAP 推定・予測・誤差計算
- `requirements.txt`: 依存関係

## setup

```shell
pip install -r requirements.txt
```

## run

```shell
python main.py
```

## what you can observe

- 真の重みと推定重みの比較
- 真のバイアスと推定バイアスの比較
- 学習データ上の MSE
- 正則化係数を大きくすると係数が縮む挙動
