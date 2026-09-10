def cubic_budget(
    step,
    total_steps,
    initial_budget,
    final_budget,
    warmup_fraction,
    final_fraction,
):

    ti = total_steps * warmup_fraction

    tf = total_steps * final_fraction

    if step < ti:
        return initial_budget

    if step > total_steps - tf:
        return final_budget

    denom = max(
        total_steps - tf - ti,
        1e-12,
    )

    progress = (
        (step - ti)
        / denom
    )

    return initial_budget - (
        initial_budget - final_budget
    ) * progress**3