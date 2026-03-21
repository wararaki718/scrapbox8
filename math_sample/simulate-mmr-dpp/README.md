# simulate-mmr-dpp

MMR (Maximal Marginal Relevance) と DPP (Determinantal Point Process) の
選択結果を比較するシミュレーションです。

## setup

```shell
pip install matplotlib numpy scikit-learn
```

## files

- `main.ipynb`: 実験・可視化の実行ノートブック
- `mmr.py`: MMR の選択アルゴリズム
- `dpp.py`: DPP のカーネル生成と greedy MAP
- `selection_export.py`: 選択過程の GIF 出力関数

## how to run

1. `main.ipynb` を上から順に実行する
2. 比較可視化セルで MMR / DPP の横並びプロットを確認する
3. 最後の出力セルで `export_selection_artifacts(...)` を実行する

## output artifacts

`export_selection_artifacts(...)` を実行すると、次が出力されます。

- `outputs/selection_process.gif`: 選択過程のアニメーション

## main parameters

- `k`: 選択件数
- `lambda_rel`: MMR の関連度重み
- `quality_scale`: DPP の quality 重み
- `fps`: GIF フレームレート
- `interval_ms`: アニメーションのフレーム間隔 (ms)
