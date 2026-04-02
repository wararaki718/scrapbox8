# dpp-diversity

Fast Greedy MAP Inference for Determinantal Point Process (DPP) の論文
"Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity"
にある Algorithm 1 / Algorithm 2 の動作を確認できる最小サンプルです。

## Files

- `fast_greedy_map.py`: Algorithm 1（高速 greedy MAP 推論）
- `fast_greedy_map_sliding_window.py`: Algorithm 2（sliding window 付き高速 greedy MAP 推論）
- `main.py`: デモ用 PSD カーネルを生成し、両アルゴリズムを実行

## Run

```bash
cd rec_sample/dpp-diversity
python main.py
```

## Notes

- カーネル行列は `L = B B^T` で構成し、PSD を満たします。
- Algorithm 1 は論文の増分更新（`c_i`, `d_i^2`）を実装しています。
- Algorithm 2 は sliding window の目的関数（論文 Eq. (10)）に基づく実装です。
