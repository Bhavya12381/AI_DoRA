class CubicBudgetScheduler:

    def __init__(
        self,
        initial_rank,
        final_rank,
        total_steps,
        warmup_fraction=0.1,
        final_fraction=0.1,
    ):

        if initial_rank < final_rank:
            raise ValueError(
                "initial_rank must be >= final_rank"
            )

        if total_steps <= 0:
            raise ValueError(
                "total_steps must be positive"
            )

        self.initial_rank = float(
            initial_rank
        )

        self.final_rank = float(
            final_rank
        )

        self.total_steps = total_steps

        self.warmup_steps = int(
            total_steps * warmup_fraction
        )

        self.final_steps = int(
            total_steps * final_fraction
        )

        self.pruning_steps = (
            total_steps
            - self.warmup_steps
            - self.final_steps
        )

    def budget(self, step):

        if step <= self.warmup_steps:
            return self.initial_rank

        pruning_end = (
            self.warmup_steps
            + self.pruning_steps
        )

        if step >= pruning_end:
            return self.final_rank

        progress = (
            step - self.warmup_steps
        ) / self.pruning_steps

        # Cubic schedule:
        #
        # starts slowly,
        # accelerates in the middle,
        # then approaches the final budget.
        reduction = progress ** 3

        return (
            self.initial_rank
            - (
                self.initial_rank
                - self.final_rank
            )
            * reduction
        )