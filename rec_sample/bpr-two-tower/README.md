# two-tower model with BPR Loss

## setup

```shell
pip install torch pytest
```

## run sample training

```shell
python main.py
```

Synthetic data 上で BPR Loss による学習を行い、学習前後の `recall@10` を表示します。

## run tests

```shell
python -m pytest -q test_bpr_two_tower.py
```
