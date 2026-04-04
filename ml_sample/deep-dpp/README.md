# deep-dpp

PyTorch implementation of **Deep Determinantal Point Processes** based on:

- Mike Gartrell, Elvis Dohmatob, Jon Alberdi. *Deep Determinantal Point Processes*.
  arXiv:1811.07245 (2018/2019).

## Research Summary

This project follows the main idea of the paper:

- Model subsets with a DPP using a low-rank kernel $L = VV^T$.
- Learn item embedding matrix $V$ using a deep feed-forward network (DeepDPP).
- Optimize the regularized log-likelihood:

$$
\mathcal{L}(V) = \sum_{n=1}^{N} \log\det(L_{A_n})
- N\log\det(L + I)
- \alpha \sum_{i=1}^{|\mathcal{Y}|} \frac{1}{\lambda_i}\|v_i\|_2^2
$$

where $A_n$ is an observed basket, $\lambda_i$ is item frequency, and $v_i$ is row $i$ of $V$.

The implementation also includes a conditional next-item scoring routine based on the
DPP Schur complement used for basket completion.

## Project Structure

- `src/deep_dpp/model.py`: DeepDPP model (MLP -> item embeddings -> DPP kernel)
- `src/deep_dpp/loss.py`: DPP objective and next-item scoring
- `src/main.py`: synthetic data demo training script
- `tests/test_deep_dpp.py`: forward/loss/training tests

## Setup

```bash
make install
```

## Run

```bash
make run
```

## Quality Check

```bash
make check
```
