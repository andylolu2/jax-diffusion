import optax


def create_optimizer(
    optimizer_type: str,
    lr_schedule: optax.Schedule,
    max_grad_norm: float,
    grac_acc_steps: int,
    **kwargs,
):
    optimizer = None
    if optimizer_type == "adam":
        optimizer = optax.adam(learning_rate=lr_schedule, **kwargs)
    else:
        raise NotImplementedError(optimizer_type)
    optimizer = optax.chain(
        optimizer,
        optax.clip_by_global_norm(max_grad_norm),
    )
    optimizer = optax.MultiSteps(optimizer, grac_acc_steps)
    return optimizer
