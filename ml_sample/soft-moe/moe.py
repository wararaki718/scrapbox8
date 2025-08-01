import einops
import torch
import torch.nn as nn

from expert import Expert, Experts


def gumbel_noise(t: torch.Tensor) -> torch.Tensor:
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -torch.log(-torch.log(noise))


class SoftMoE(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_data: int,
        num_experts: int = 4,
        num_slots: int | None = None,
        dropout: float = 0.,
        use_layernorm: bool = False
    ):
        super().__init__()
        
        self._n_data = n_data
        self._num_experts = num_experts
        if num_slots is None:
            self._num_slots = n_data // num_experts
        else:
            self._num_slots = num_slots

        assert self._n_data == self._num_slots * self._num_experts, 'n_data must be a multiple of num_slots * num_experts'

        norm_klass = nn.LayerNorm if use_layernorm else nn.RMSNorm

        self.norm = norm_klass(n_input)
        self.slot_norm = norm_klass(n_input)
        self.slot_embeds = nn.Parameter(torch.randn(self._num_experts, self._num_slots, n_input))

        expert_list = [Expert(n_input=n_input, n_output=n_output, dropout=dropout) for _ in range(self._num_experts)]
        self.experts = Experts(expert_list)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        add_noise: bool = False,
        noise_mult: float = 1.0,
    ) -> torch.Tensor:
        # 1. Normalization
        x = self.norm(x)
        slot_embeds = self.slot_norm(self.slot_embeds)

        # 2. Compute logits
        logits = torch.einsum('b d, e s d -> b e s', x, slot_embeds)

        # 3. Apply noise and mask
        if add_noise:
            logits += gumbel_noise(logits) * noise_mult
        
        if mask is not None:
            mask = einops.rearrange(mask, 'b -> b 1 1')
            logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

        # 4. Compute dispatch and combine weights
        dispatch_weights = logits.softmax(dim=1)
        combine_weights = einops.rearrange(logits, 'b e s -> b (e s)').softmax(dim=-1)

        # 5. Create slots
        slots = torch.einsum('b d, b e s -> b e s d', x, dispatch_weights)

        # 6. Process slots with experts
        out = self.experts(slots)

        # 7. Combine back the output
        out = einops.rearrange(out, 'b e s d -> b (e s) d')
        out = torch.einsum('b s d, b s -> b d', out, combine_weights)

        return out