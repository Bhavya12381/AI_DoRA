class CubicBudgetScheduler:
    """
    Cubic schedule for the target average adaptive rank.

    The pruning phase occupies a configurable portion of training.
    Before pruning starts, the target remains at the initial rank.
    During pruning, the target rank decreases cubically.
    After pruning ends, the target remains at the final rank.
    """

    def __init__(
        self,
        initial_rank,
        final_rank,
        total_steps,
        start_fraction=0.15,
        end_fraction=0.50,
    ):
        if initial_rank < final_rank:
            raise ValueError(
                "initial_rank must be >= final_rank"
            )

        if total_steps <= 0:
            raise ValueError(
                "total_steps must be positive"
            )

        if not 0.0 <= start_fraction < end_fraction <= 1.0:
            raise ValueError(
                "start_fraction and end_fraction must satisfy "
                "0 <= start_fraction < end_fraction <= 1"
            )

        self.initial_rank = float(initial_rank)
        self.final_rank = float(final_rank)
        self.total_steps = int(total_steps)

        self.start_step = int(
            self.total_steps * start_fraction
        )
        self.end_step = int(
            self.total_steps * end_fraction
        )

        if self.end_step <= self.start_step:
            raise ValueError(
                "pruning phase must contain at least one step"
            )

    def budget(self, step):
        if step <= self.start_step:
            return self.initial_rank

        if step >= self.end_step:
            return self.final_rank

        progress = (
            (step - self.start_step)
            / (self.end_step - self.start_step)
        )

        cubic_progress = progress ** 3

        return (
            self.initial_rank
            - (self.initial_rank - self.final_rank)
            * cubic_progress
        )