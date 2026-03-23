import numpy as np


def main() -> None:
    # 1. 質のスコア (q): アイテムごとの「重み」や「関連度スコア」
    q = np.array([0.9, 0.8, 0.7, 0.6])

    # 2. 特徴行列 (Phi): 各アイテムの埋め込みベクトル (N x d)
    # 例えば、ニュース記事のトピックベクトルなどを想定
    Phi = np.array([
        [1.0, 0.1],  # アイテム0
        [0.9, 0.2],  # アイテム1
        [0.1, 0.9],  # アイテム2
        [0.2, 0.8]   # アイテム3
    ])

    # 3. カーネル行列 L の生成
    # 式: L_{ij} = q_i * (phi_i・phi_j) * q_j
    # これは行列演算で Phi @ Phi.T と q[:, None] @ q[None, :] の要素ごとの積で表現できます
    similarity = Phi @ Phi.T
    L = np.outer(q, q) * similarity

    print("行列 L:")
    print(np.round(L, 2))


if __name__ == "__main__":
    main()
