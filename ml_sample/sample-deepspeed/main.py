import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

from model import Model
from utils import get_data


def main() -> None:
    model = Model()
    print("model defined!")

    data = get_data()
    print(data.shape)

    # model defined
    parameters = {"params": [p for p in model.parameters()], "name": "parameters"}
    model_parameters = split_params_into_different_moe_groups_for_optimizer(parameters)
    ds_config = {
        "train_batch_size": 16,
        "steps_per_print": 2000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },
    }

    args = {
        "local_rank": 1,
    }
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        training_data=data,
        config=ds_config,
    )

    model_engine.eval()
    with torch.no_grad():
        output = model_engine(data)
    print(output.shape)
    print("DONE")


if __name__ == "__main__":
    main()
