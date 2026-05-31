import torch


def load_data() -> torch.Tensor:
    R = torch.tensor([
        [3.,0.,1.,0.],
        [0.,2.,0.,1.],
        [1.,0.,4.,2.]
    ])
    return R
