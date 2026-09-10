import torch
from torch import nn


class DoRALinear(nn.Module):

    def __init__(
        self,
        base: nn.Linear,
        start_rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if start_rank < 1:
            raise ValueError("start_rank must be >= 1")

        self.base = base
        self.start_rank = start_rank
        self.alpha = alpha
        self.scaling = alpha / start_rank
        self.dropout = nn.Dropout(dropout)

        self.base.weight.requires_grad_(False)

        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(
            torch.empty(start_rank, base.in_features)
        )

        self.lora_B = nn.Parameter(
            torch.empty(base.out_features, start_rank)
        )

        self.c = nn.Parameter(
            torch.ones(start_rank)
        )

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=5**0.5
        )

        nn.init.kaiming_uniform_(
            self.lora_B,
            a=5**0.5
        )

    def delta_weight(self):

        scaled_A = self.lora_A * self.c[:, None]

        return self.lora_B @ scaled_A * self.scaling

    def component_weights(self):

        return torch.einsum(
            "or,ri->roi",
            self.lora_B,
            self.lora_A * self.c[:, None],
        ) * self.scaling

    def forward(self, x):

        base_out = self.base(x)

        z = self.dropout(x)

        z = torch.matmul(
            z,
            self.lora_A.t()
        )

        z = z * self.c

        update = torch.matmul(
            z,
            self.lora_B.t()
        )

        update = update * self.scaling

        return base_out + update