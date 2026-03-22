# DPP の説明

このドキュメントは、現在の実装（`dpp/algorithm.py`, `dpp/visualization.py`, `dpp/map_simulation.py`）に対応した
DPPの概要と、MAP推論・Cholesky・rank-1 updateの関係をまとめたものです。

## 1. DPPアルゴリズムの概要

Determinantal Point Process (DPP) は、
「品質が高く、かつ互いに似すぎていない（多様な）要素集合」を選ぶための確率モデルです。

本実装では L-ensemble 形式を使っており、カーネル行列を $L$ とすると、
部分集合 $S$ のスコアは次で表されます。

$$
\text{score}(S) \propto \det(L_S)
$$

- $L_S$: $L$ を $S$ の行・列で切り出した部分行列
- 行列式が大きいほど、品質と多様性の両方の観点で望ましい集合

実装では数値安定化のために、対数を取った目的関数を使っています。

$$
f(S) = \log\det(L_S)
$$

## 1.1 L-ensemble とは何か

L-ensemble は、DPPを「カーネル行列 $L$ で直接定義する」表現です。
有限集合の任意の部分集合 $S$ に対して、重みを

$$
w(S) = \det(L_S)
$$

で与えます。これを正規化すると確率分布になります。

$$
P(S) = \frac{\det(L_S)}{\det(I + L)}
$$

ここで:

- $L$ は対称な半正定値行列（PSD）
- $L_S$ は $S$ で切り出した主小行列
- 分母 $\det(I+L)$ は全部分集合の重みを足し合わせた正規化定数

直感的には、$\det(L_S)$ は「品質」と「多様性」の両方を反映します。
似た要素ばかりの集合では行列の列/行が近くなり、行列式が小さくなりやすいです。

本実装との対応:

- `build_l_kernel` で L-ensemble の $L$ を構築
- 類似度はRBFカーネル、品質は `quality` ベクトルで付与
- 実装形は概ね

$$
L_{ij} = q_i\, s_{ij}\, q_j
$$

（$q_i$: quality, $s_{ij}$: 類似度）

## 2. MAP Inference（Greedy）の概要

MAP推論は「最もスコアの高い集合 $S$」を探す問題です。
厳密最適化は一般に難しいため、本実装は greedy 近似を使っています。

各ステップで、未選択候補 $i$ に対し次を計算します。

$$
\Delta_i = \log\det\left(L_{S\cup\{i\}}\right) - \log\det\left(L_S\right)
$$

この $\Delta_i$（marginal gain）が最大の候補を選択します。

- marginal gain が大きい: その候補を追加したときの目的関数改善が大きい
- greedy MAP: 毎ステップで最大の $\Delta_i$ を選ぶ
- 実装オプションで、改善が非正なら停止できます

## 2.1 アルゴリズムのステップ

本実装の greedy MAP 推論は、次の流れで進みます。

1. 点群から類似度行列を作り、quality を掛けて $L$ を構築する
2. 現在の選択集合 $S$ に対して、各候補 $i$ の marginal gain $\Delta_i$ を計算する
3. $\Delta_i$ が最大の候補を1つ選び、$S$ に追加する
4. 追加後の $L_S$ から Cholesky を計算し、可視化用に保存する
5. 同時に Schur 補の rank-1 update を行い、未選択側の残差行列を更新する
6. 最大ステップ数に達するか、停止条件を満たすまで 2-5 を繰り返す

## 3. marginal gain について

marginal gain は、
現在の選択集合 $S$ に候補 $i$ を1つ追加したときの、
DPP目的関数 $\log\det(L_S)$ の増分を表します。

$$
\Delta_i = \log\det\left(L_{S\cup\{i\}}\right) - \log\det\left(L_S\right)
$$

直感的には「次に1個追加する価値」を示す指標です。

## 4. rank-1 update行列とCholesky行列の関係について

この実装では、rank-1 update行列とCholesky行列は
同じステップの異なる視点を表します。

- rank-1 update 側（残差行列）:
  - 選んだ要素（pivot）を使って Schur 補行列を rank-1 で更新
  - その後、選ばれた要素の行・列を除去
  - よって `rank1_matrix` は「未選択要素だけ」の空間を表す

- Cholesky 側（selected 部分行列）:
  - `chol_sub` は毎ステップ、元の $L$ から
    選択集合 $S$ の部分行列 $L_S$ を切り出して再計算
  - つまり `rank1_matrix` から直接Choleskyしているわけではない

要するに、同じステップで2つの見方を保持しています。

1. selected側の見方: $L_S$ とその Cholesky (`chol_sub`)
2. remaining側の見方: rank-1 update 後の残差行列 (`rank1_matrix`)

## 5. Cholesky の振る舞い

$L_S$ はDPPカーネルの部分行列で、対称正定値に近い構造を持つため、
Cholesky分解

$$
L_S = R R^\top
$$

（または実装形に応じて下三角 $LL^\top$）が可能です。

実装では、ステップごとに選択集合が1要素増えるため、
`chol_sub` のサイズも $|S| \times |S|$ で拡大していきます。

可視化上は「選択済みサブシステムの内部構造の成長」を確認するための行列です。

## 6. rank-1 update（Schur）の振る舞い

選択した pivot を中心に Schur 補行列を rank-1 更新し、
次ステップの候補空間（未選択要素のみ）へ縮約します。

この処理によって、次ステップでの評価対象は
「既に選択済みの影響を織り込んだ残差空間」になります。

可視化上の `rank1_matrix` は、
ステップが進むほど次元が小さくなっていくのが特徴です。

## 6.1 Schur 補行列とは何か

Schur 補行列は、ブロック行列の一部を消去した後に残る
「補正込みの有効行列」です。ガウス消去の行列表現とみなせます。

ブロック行列

$$
M=
\begin{bmatrix}
A & B \\
C & D
\end{bmatrix}
$$

について、$A$ が可逆なら $A$ に関する Schur 補は

$$
M/A = D - C A^{-1} B
$$

です（同様に $D$ が可逆なら $M/D = A - B D^{-1} C$）。

本実装での対応:

- pivot（選択された要素）を中心に rank-1 update

$$
S' = S - \frac{k k^\top}{p}
$$

を計算し、これは Schur 補の1ステップ更新に対応します。
- その後、pivot の行・列を削除して次ステップの残差行列に進みます。
- 結果として `rank1_matrix` は
  「選択済み要素の影響を織り込んだ未選択側の有効カーネル」を表します。

## 7. 実装との対応

- DPPコア: `dpp/algorithm.py`
  - `build_l_kernel`
  - `greedy_map_with_trace`
  - `logdet_subset`
- 描画: `dpp/visualization.py`
  - `render_dpp_frame`
  - `animate_dpp_trace`
- 実行エントリ: `dpp/map_simulation.py`
  - `run_simulation`
