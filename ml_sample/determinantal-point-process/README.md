# determinantal-point-process

NumPy で DPP (Determinantal Point Process) を扱う最小実装です。
サンプリングと MAP 推論（greedy 近似）を切り替えて実行できます。
デモ入力だけでなく、特徴量行列または L-kernel 行列をファイルから読み込んで再利用できます。

## 実行方法

```bash
python main.py
```

オプション例:

```bash
python main.py --n-items 30 --n-features 12 --temperature 0.7 --seed 123
```

特徴量行列から実行:

```bash
python main.py --features-path data/features.csv --mode map --max-length 10
```

L-kernel 行列から実行:

```bash
python main.py --kernel-path data/kernel.npy --mode sample --seed 123
```

MAP 推論（greedy + Cholesky 更新系）:

```bash
python main.py --mode map --max-length 8
```

## 実装内容

- 特徴量行列から PSD な L-ensemble カーネルを構築
- `.npy` / `.csv` / `.tsv` / `.txt` から特徴量行列または L-kernel をロード可能
- 固有値分解に基づく標準的な DPP サンプリング
- greedy に logdet を最大化する DPP MAP 推論（近似）
- 選択されたインデックス集合を表示

## 補足

- `sample` は確率的サンプリングです
- `map` は greedy 近似であり、厳密 MAP ではありません
- `map` は追加利得が正のときだけ要素を追加するため、`max-length` 未満で停止することがあります
- `sample_dpp` と `map_inference_dpp` は対称な L-kernel を前提とします
- `--features-path` と `--kernel-path` は同時指定できません


