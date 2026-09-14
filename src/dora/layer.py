import torch
from torch import nn


class AdaptiveRankLinear(nn.Module):
    """
    Linear layer with a frozen base transformation and a trainable
    low-rank residual made from individual rank-1 components.

    The residual is:

        Delta W = sum_i c_i * b_i outer a_i

    where:
        a_i : input-side vector
        b_i : output-side vector
        c_i : trainable scalar gate
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError("rank must be positive")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Keep the original linear transformation.
        self.base = nn.Linear(
            self.in_features,
            self.out_features,
            bias=base_layer.bias is not None,
        )

        self.base.weight.data.copy_(base_layer.weight.data)

        if base_layer.bias is not None:
            self.base.bias.data.copy_(base_layer.bias.data)

        # The pretrained/base parameters are frozen.
        self.base.weight.requires_grad = False

        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # Each row of A represents one input-side vector.
        #
        # Shape:
        #     [rank, in_features]
        self.A = nn.Parameter(
            torch.empty(
                rank,
                self.in_features,
            )
        )

        # Each column of B represents one output-side vector.
        #
        # Shape:
        #     [out_features, rank]
        self.B = nn.Parameter(
            torch.empty(
                self.out_features,
                rank,
            )
        )

        # One scalar gate per rank component.
        #
        # Shape:
        #     [rank]
        self.c = nn.Parameter(
            torch.zeros(rank)
        )

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

        # Once a component is pruned, this mask prevents it
        # from becoming active again.
        self.active_mask: torch.Tensor
        self.register_buffer(
            "active_mask",
            torch.ones(rank),
        )

    def reset_parameters(self):
        """
        Initialize the two low-rank factors.

        We initialize both factors with Kaiming initialization
        because each rank component must start with a meaningful
        direction before importance-based pruning begins.
        """

        nn.init.kaiming_uniform_(
            self.A,
            a=5 ** 0.5,
        )

        nn.init.kaiming_uniform_(
            self.B,
            a=5 ** 0.5,
        )

    def forward(self, x):
        """
        Compute:

            base(x) + low_rank_update(x)
        """

        base_output = self.base(x)

        # Apply dropout only to the adapter input.
        adapter_input = self.dropout(x)

        # A maps input features to rank components.
        #
        # [batch, in_features]
        #        ↓ A
        # [batch, rank]
        low_rank = adapter_input @ self.A.T

        # Apply the scalar gate belonging to each component.
        #
        # [batch, rank]
        #        *
        # [rank]
        low_rank = low_rank * (
            self.c * self.active_mask
        )

        # B maps rank components back to output space.
        #
        # [batch, rank]
        #        ↓ B
        # [batch, out_features]
        low_rank = low_rank @ self.B.T

        low_rank = low_rank * self.scale

        return base_output + low_rank

    def component_matrices(self):
        """
        Return every rank-1 weight update separately.

        Component i is:

            c_i * b_i outer a_i

        Returned shape:

            [rank, out_features, in_features]
        """

        scaled_A = (
            self.A
            * (self.c * self.active_mask).unsqueeze(1)
        )

        return torch.einsum(
            "or,ri->roi",
            self.B,
            scaled_A,
        )

    def merged_update(self):
        """
        Return the complete low-rank weight update:

            Delta W = sum_i Delta W_i

        Shape:

            [out_features, in_features]
        """

        return self.component_matrices().sum(dim=0)

    def prune_components(self, indices):
        """
        Permanently disable selected rank components.

        We do not physically delete their parameters.

        Instead, their gate is forced to zero and the active mask
        prevents them from contributing again.
        """

        if len(indices) == 0:
            return

        with torch.no_grad():

            for index in indices:

                self.c[index] = 0.0
                self.active_mask[index] = 0.0

    def active_rank(self):
        """
        Number of currently active rank components.
        """

        return int(
            self.active_mask.sum().item()
        )


class DoRALinear(AdaptiveRankLinear):
    """Compatibility wrapper for the original DoRA layer API."""

    def __init__(
        self,
        base_layer: nn.Linear,
        start_rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__(
            base_layer,
            rank=start_rank,
            alpha=alpha,
            dropout=dropout,
        )

    @property
    def start_rank(self):
        return self.rank

    @property
    def lora_A(self):
        return self.A

    @property
    def lora_B(self):
        return self.B

    def component_weights(self):
        return self.component_matrices()

    def delta_weight(self):
        return self.merged_update()