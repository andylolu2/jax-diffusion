import optax


def create_lr_schedule(schedule_type: str, **kwargs):
    if schedule_type == "constant":
        return optax.constant_schedule(**kwargs)
    elif schedule_type == "constant_warmup":
        return _constant_with_warmup(**kwargs)
    elif schedule_type == "cosine":
        return _cosine_with_warmup(**kwargs)
    else:
        raise NotImplementedError(schedule_type)


def _constant_with_warmup(value: float, warmup_steps: int):
    warmup = optax.linear_schedule(0, value, warmup_steps)
    constant = optax.constant_schedule(value=value)
    return optax.join_schedules([warmup, constant], boundaries=[warmup_steps])


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
        end_value=peak_value / decay_factor,
    )
