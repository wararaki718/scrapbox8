# dppnet

PyTorch sample implementation of:

- Zelda Mariet, Yaniv Ovadia, Jasper Snoek.
	*DPPNet: Approximating Determinantal Point Processes with Deep Networks*.
	https://arxiv.org/abs/1901.02051

This project implements a compact DPPNet-style model for a fixed-size ground set,
including:

- Inhibitive attention from Eq. (4)
- Conditional marginals target from Eq. (2)
- Iterative sampling / greedy mode from Algorithm 1

## What This Sample Covers

- `src/dppnet/model.py`
	- DPPNet MLP that predicts item-wise conditional inclusion probabilities.
	- Inhibitive attention:
		- Compute dissimilarity per selected item as `1 - softmax(Q Phi^T / sqrt(d))`
		- Aggregate by element-wise product over selected items
		- Normalize to a simplex vector `a`

- `src/dppnet/dpp.py`
	- Exact conditional marginals from the DPP kernel:
		- `v_i = 1 - [(L + I_{S^c})^{-1}]_{ii}`
	- RBF-kernel builder for synthetic experiments
	- DPPNet subset sampling (stochastic and greedy)

- `src/main.py`
	- End-to-end demo:
		- builds synthetic features and an RBF DPP kernel
		- trains DPPNet to regress exact conditional marginals
		- samples subsets with conditioning

- `tests/test_dppnet.py`
	- Attention simplex and shape checks
	- Conditional marginals correctness constraints
	- Forward-pass mask checks
	- Training-loss decrease test
	- Sampler conditioning/cardinality test

## Setup

```bash
make install
```

## Run Demo

```bash
make run
```

## Quality Check

```bash
make check
```

## Notes

- This is a compact educational approximation, not an exact reproduction of all
	experimental setups in the paper (MNIST/CelebA/MovieLens pipelines are omitted).
- The model is trained with supervised targets from exact DPP conditional
	marginals to mimic DPP behavior efficiently at inference time.
