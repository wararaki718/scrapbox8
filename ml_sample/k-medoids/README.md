# sample k-medoids

Pythonでk-medoidsアルゴリズム（ユークリッド距離）の最小実装です。

## 実行方法

```bash
pip install numpy
python main.py
```

## 内容

- `main.py`
  - サンプル2次元データを用意
  - PAMに近い単純なswap探索でmedoidを更新
  - 最終的なmedoid, クラスタ割当, コストを表示
