# DPP Fast Greedy MAP Inference Notes

このメモは、論文
"Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity"
にある Algorithm 1 と Algorithm 2 のロジックを実装視点で整理したものです。

## 前提

- 対象は DPP のカーネル行列 `L`（対称 PSD 行列）
- 目的は、行列式 `det(L_Y)` を大きくするようにアイテム集合 `Y` を greedy に構築すること
- greedy の 1 ステップで必要なのは、候補 `i` を追加したときの増分

  `log det(L_{Y ∪ {i}}) - log det(L_Y)`

---

## Algorithm 1: Fast Greedy MAP Inference

### アイデア

Cholesky 分解の増分更新を使って、各候補の増分を低コストで更新する。

- 追加候補 `i` に対して
  - ベクトル `c_i`
  - スカラー `d_i^2`
  を持つ
- 増分は `log(d_i^2)` になるため、毎回 `d_i^2` 最大の候補を選べばよい

### 初期化

- `Y = ∅`
- 各候補について
  - `c_i = []`
  - `d_i^2 = L_ii`

### 反復

1. 未選択候補の中から `d_i^2` が最大の `j` を選ぶ
2. `j` を `Y` に追加
3. 残り候補 `i` について更新
   - `e_i = (L_ji - <c_j, c_i>) / d_j`
   - `c_i <- [c_i, e_i]`
   - `d_i^2 <- d_i^2 - e_i^2`
4. 停止条件を満たしたら終了

### 停止条件

- 固定長 Top-N: `|Y| = N`
- 無制約: best `d_j^2 < 1`（論文）
- 実装上は数値安定のため `d_j^2 <= epsilon` でも停止

### 計算量

- 無制約 MAP: `O(M^3)`
- Top-N: `O(N^2 M)`

---

## Algorithm 2: Fast Greedy MAP with Sliding Window

### 目的

長い推薦列では「全体」ではなく「直近の近傍」だけ多様であればよい場合がある。
そこで、直近 `w-1` 件だけを対象に増分を計算する。

### 目的関数

- `Y_w`: 現在の出力列のうち、直近 `w-1` 件
- 次に選ぶ `j` は以下を最大化

  `log det(L_{Y_w ∪ {j}}) - log det(L_{Y_w})`

### 反復の流れ

1. 現在の `Y_w` を構成
2. 未選択候補ごとに上式の増分を計算
3. 最大の候補を選択
4. 追加後、窓サイズ `w` を超えたら最古要素を窓から外す
5. 必要件数まで繰り返す

### 特徴

- 近傍多様性を制御できる（`w` が小さいほど局所的）
- 論文の実装では Cholesky の更新・削除を in-place で行い高速化
- 論文上の計算量は `O(w N M)`

---

## Algorithm 1 と Algorithm 2 の使い分け

- 全体の組合せ多様性を重視: Algorithm 1
- 長い推薦列で局所多様性を重視: Algorithm 2（window を設定）

実運用では、候補数 `M`、返却件数 `N`、表示 UI の閲覧幅に対応する window `w` の3つで選択するとよい。
