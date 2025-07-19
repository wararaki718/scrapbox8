import torch

from models.moe import MixtureOfExperts, DeepMixtureOfExperts, DistributedMixtureOfExperts
from loss import loss_per_token


def main() -> None:
    # define parameters
    batch_size = 2
    sequence_length = 256
    
    n_input = 512
    n_hidden = 512 * 4
    n_experts = 8
    n_classes = 1024

    x = torch.randn(batch_size, sequence_length, n_input)
    labels = torch.randint(0, n_classes, (batch_size, sequence_length))

    # create model
    model = MixtureOfExperts(
        n_experts=n_experts,
        n_input=n_input,
        n_hidden=n_hidden,
        n_output=n_classes,
    )
    deep_model = DeepMixtureOfExperts(
        n_experts=n_experts,
        n_input=n_input,
        n_hidden=n_hidden,
        n_output=n_classes,
        depth=3,
    )

    print("## model")
    y, probs, output = model(x)
    print(y.shape, probs.shape, output.shape)

    loss = loss_per_token(probs, output, labels)
    print(f"Output shape: {output.shape}")
    print(f"Loss shape: {loss.shape}")
    print()

    print("## deep model")
    y_deep, probs_deep, output_deep = deep_model(x)
    print(y_deep.shape, probs_deep.shape, output_deep.shape)

    loss_deep = loss_per_token(probs_deep, output_deep, labels)
    print(f"Output shape: {output_deep.shape}")
    print(f"Loss shape: {loss_deep.shape}")
    print()

    # print("## distributed model")
    # if not torch.distributed.is_initialized():
    #     torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # distributed_model = DistributedMixtureOfExperts(
    #     n_input=n_input,
    #     n_hidden=n_hidden,
    #     n_output=n_classes,
    #     k=2,
    #     capacity_factor=1.25,
    #     padding_val=0,
    # )
    # y_dist, probs_dist, output_dist = distributed_model(x)
    # print(y_dist.shape, probs_dist.shape, output_dist.shape)
    # loss_dist = loss_per_token(probs_dist, output_dist, labels)
    # print(f"Output shape: {output_dist.shape}")
    # print(f"Loss shape: {loss_dist.shape}")
    # print()
    # torch.distributed.barrier()
    # torch.distributed.destroy_process_group()

    print("DONE")

if __name__ == "__main__":
    main()
