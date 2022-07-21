import optax


def create_lr_schedule(schedule_type: str, **kwargs):
    if schedule_type == "constant":
        return optax.constant_schedule(**kwargs)
    elif schedule_type == "cosine":
        return _cosine_with_warmup(**kwargs)
    else:
        raise NotImplementedError(schedule_type)


def _cosine_with_warmup(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    decay_factor: float,
):
    return optax.warmup_cosine_decay_schedule(
        init_value=init_value,
        peak_value=peak_value,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=peak_value * decay_factor,
    )


def create_optimizer(
    optimizer_type: str,
    lr_schedule: optax.Schedule,
    max_grad_norm: float,
    **kwargs,
):
    optimizer = None
    if optimizer_type == "adam":
        optimizer = optax.adam(learning_rate=lr_schedule, **kwargs)
    else:
        raise NotImplementedError(optimizer_type)
    return optax.chain(
        optimizer,
        optax.clip_by_global_norm(max_grad_norm),
    )
