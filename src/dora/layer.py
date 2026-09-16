import torch
from torch import nn


class AdaptiveRankLinear(nn.Module):
    """
    Linear layer with a frozen base transformation and a trainable
    low-rank residual composed of individual rank-1 components.

    The residual is:

        Delta W = sum_i c_i * (b_i outer a_i)

    where:

        a_i : input-side vector
        b_i : output-side vector
        c_i : trainable scalar gate

    A has shape:

        [rank, in_features]

    B has shape:

        [out_features, rank]

    c has shape:

        [rank]

    The effective weight update is:

        Delta W = B @ diag(c) @ A

    followed by the LoRA scaling factor alpha / rank.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                "base_layer must be an instance of torch.nn.Linear"
            )

        if rank <= 0:
            raise ValueError("rank must be positive")

        if alpha < 0:
            raise ValueError("alpha must be non-negative")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # ------------------------------------------------------------
        # Frozen base linear transformation
        # ------------------------------------------------------------

        self.base = nn.Linear(
            self.in_features,
            self.out_features,
            bias=base_layer.bias is not None,
        )

        with torch.no_grad():
            self.base.weight.copy_(base_layer.weight)

            if base_layer.bias is not None:
                self.base.bias.copy_(base_layer.bias)

        self.base.weight.requires_grad = False

        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # ------------------------------------------------------------
        # Trainable adaptive-rank parameters
        # ------------------------------------------------------------

        # One input-side vector per rank component.
        #
        # Shape:
        #     [rank, in_features]
        self.A = nn.Parameter(
            torch.empty(
                rank,
                self.in_features,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        )

        # One output-side vector per rank component.
        #
        # Shape:
        #     [out_features, rank]
        self.B = nn.Parameter(
            torch.empty(
                self.out_features,
                rank,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        )

        # One scalar gate for every rank-1 component.
        #
        # Shape:
        #     [rank]
        #
        # Starting at zero means the adapter initially contributes
        # no residual update to the pretrained model.
        self.c = nn.Parameter(
            torch.zeros(
                rank,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        )

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the trainable low-rank parameters.

        A and B use Kaiming-uniform initialization.

        c starts at zero so the initial adapter contribution is zero.
        """

        nn.init.kaiming_uniform_(
            self.A,
            a=5 ** 0.5,
        )

        nn.init.kaiming_uniform_(
            self.B,
            a=5 ** 0.5,
        )

        with torch.no_grad():
            self.c.zero_()

    def forward(self, x):
        """
        Compute:

            base(x) + low_rank(x)

        where:

            low_rank(x)
                = dropout(x)
                -> A
                -> component gates c
                -> B
                -> alpha / rank
        """

        base_output = self.base(x)

        adapter_input = self.dropout(x)

        # [batch, ..., in_features]
        # ->
        # [batch, ..., rank]
        low_rank = adapter_input @ self.A.T

        # Apply one scalar gate to each rank component.
        low_rank = low_rank * self.c

        # [batch, ..., rank]
        # ->
        # [batch, ..., out_features]
        low_rank = low_rank @ self.B.T

        # LoRA scaling.
        low_rank = low_rank * self.scale

        return base_output + low_rank

    def component_matrices(self):
        """
        Return every rank-1 weight component separately.

        Returns:

            [rank, out_features, in_features]

        Component i is:

            c_i * (b_i outer a_i)

        before the global alpha / rank scaling.

        The scaling is intentionally applied here so that the
        returned components correspond to the actual weight update.
        """

        scaled_A = self.A * self.c.unsqueeze(1)

        components = torch.einsum(
            "or,ri->roi",
            self.B,
            scaled_A,
        )

        return components * self.scale

    def merged_update(self):
        """
        Return the complete low-rank weight update.

        Shape:

            [out_features, in_features]
        """

        return self.component_matrices().sum(dim=0)

    def prune_components(self, indices):
        """
        Set selected component gates to zero.

        The A and B parameters are retained.

        This is important because a component that has been pruned
        can potentially become active again if its scalar gate c
        becomes non-zero during subsequent optimization.
        """

        if len(indices) == 0:
            return

        with torch.no_grad():
            for index in indices:
                if index < 0 or index >= self.rank:
                    raise IndexError(
                        f"component index {index} is outside "
                        f"[0, {self.rank})"
                    )

                self.c[index] = 0.0

    def active_rank(self):
        """
        Return the number of components whose scalar gates are
        currently non-zero.
        """

        return int(
            torch.count_nonzero(
                self.c.detach()
            ).item()
        )


class DoRALinear(AdaptiveRankLinear):
    """
    Compatibility wrapper around AdaptiveRankLinear.

    The name DoRALinear is retained so existing tests and imports
    continue to work.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        start_rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__(
            base_layer=base_layer,
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