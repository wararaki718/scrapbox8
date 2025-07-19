import torch
import torch.nn as nn

from .router import Router
from .nn import NeuralNetwork


class MixtureOfExperts(nn.Module):
    def __init__(self, n_experts: int, n_input: int, n_hidden: int, n_output: int) -> None:
        super().__init__()

        self._router = Router(n_experts, n_input, n_hidden)
        self._experts = nn.ModuleList(
            [NeuralNetwork(n_input, n_hidden, n_output) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        routing_weights = self._router(x)
        expert_outputs = torch.stack([expert(x) for expert in self._experts], dim=-2)
        weighted_outputs = expert_outputs * routing_weights.unsqueeze(-1)
        return weighted_outputs.sum(dim=-2), routing_weights, expert_outputs


class DeepMixtureOfExperts(nn.Module):
    def __init__(self, n_experts: int, n_input: int, n_hidden: int, n_output: int, depth: int) -> None:
        super().__init__()
        experts = [
            MixtureOfExperts(n_experts, n_input, n_hidden, n_input)
            for _ in range(depth - 1)
        ]
        experts.append(MixtureOfExperts(n_experts, n_input, n_hidden, n_output))
        self._model = nn.Sequential(*experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for layer in self._model:
            x, routing_weights, expert_outputs = layer(x)
        return x, routing_weights, expert_outputs


class DistributedMixtureOfExperts(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        k: int=2,
        capacity_factor: float=1.25,
        padding_val: int=0,
    ) -> None:
        super().__init__()
        self._k = k
        self._capacity_factor = capacity_factor
        self._padding_val = padding_val

        self._n_experts = torch.distributed.get_world_size()
        self._router = torch.nn.parallel.DistributedDataParallel(
            Router(self._n_experts, k, capacity_factor, padding_val)
        )
        self._expert = FeedForward(
            n_input, n_hidden, n_output,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B,T,C = x.shape
    
        probs = self._router(x)
        topk_probs, topk_experts = torch.topk(probs, k=self._k)
        ids_per_expert = [(topk_experts==expert).nonzero() for expert in range(self._n_experts)]
        probs_per_expert = [topk_probs[topk_experts==expert] for expert in range(self._n_experts)]

        # all-to-all to exchange the count of inputs to send/receive to/from each processor
        send_count = [torch.tensor([len(ids)], dtype=torch.int64) for ids in ids_per_expert]
        recv_count = [torch.tensor([0], dtype=torch.int64) for _ in ids_per_expert]
        dist.all_to_all(recv_count, send_count)
        fn_count = lambda tensor, scale=1: [x.item() * scale for x in tensor]

        # send/receive the metadata row_id+token_id to/from the appropriate processors
        M = ids_per_expert[0].shape[-1]
        send_ids = torch.cat(ids_per_expert, dim=0)
        send_ids[:,0] += global_rank * B
        recv_ids = torch.zeros(sum(recv_count) * M, dtype=send_ids.dtype)
        dist.all_to_all_single(recv_ids, send_ids.flatten(), fn_count(recv_count, M), fn_count(send_count, M))
        recv_ids = recv_ids.view(-1, M)

        # send/receive input tokens to/from the appropriate processors
        send_toks = torch.cat([x[ids[:, :2].T.tolist()] for ids in ids_per_expert], dim=0)
        recv_toks = torch.zeros(sum(recv_count) * C, dtype=x.dtype)
        dist.all_to_all_single(recv_toks, send_toks.flatten(), fn_count(recv_count, C), fn_count(send_count, C))
        recv_toks = recv_toks.view(-1, C) # reshape to C columns

        # group received metadata by row id
        uniq_rows, recv_row_lens = recv_ids[:, 0].unique(sorted=True, return_counts=True)
        recv_row_offsets = [0] + torch.cumsum(recv_row_lens, dim=0).tolist()
        recv_row_slice = lambda row: slice(recv_row_offsets[row], recv_row_offsets[row+1])

        # crop or pad received items PER SENTENCE to max capacity. Batch shape: Rows * Capacity * C
        capacity = int(T / self._n_experts * self._capacity_factor)
        pad_fn = lambda toks, value = self._padding_val: F.pad(toks, (0, 0, 0,capacity - toks.shape[0]), value=value)
        batch_toks = torch.stack([pad_fn(recv_toks[recv_row_slice(i)]) for i in range(len(uniq_rows))], dim=0)

        batch_row_len = torch.tensor([min(recv_row_lens[r], capacity) for r in range(len(uniq_rows))])
        batch_toks = self._expert(batch_toks)

        recv_toks = recv_toks.fill_(self._padding_val)
        recv_tok_offsets  = np.concatenate([range(recv_row_offsets[i], recv_row_offsets[i]+batch_row_len[i]) for i in range(len(uniq_rows))])
        batch_tok_offsets = np.concatenate([[[i]*batch_row_len[i], range(batch_row_len[i])] for i in range(len(uniq_rows))], axis=1)
        recv_toks[recv_tok_offsets] = batch_toks[batch_tok_offsets[0], batch_tok_offsets[1]]

        send_toks = send_toks.fill_(self._padding_val).flatten()
        dist.all_to_all_single(send_toks, recv_toks.flatten(), fn_count(send_count,C), fn_count(recv_count,C))
        x = send_toks.view(-1,C)
        x *= torch.cat(probs_per_expert).view(-1,1)
        if self._k > 1:
            x = torch.stack([x[send_ids[:,-1]==k] for k in range(self._k)]).sum(dim=0)
        return x.view(B,T,C)
