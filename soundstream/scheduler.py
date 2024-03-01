"""Scheduler Factory"""
import logging
import math

__all__ = ["create_scheduler"]

_logger = logging.getLogger(__name__)


def linear_lr(start_factor, end_factor, total_iters, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    start_lr = lr * start_factor
    end_lr = lr * end_factor
    for i in range(steps):
        epoch_idx = math.floor(i / steps_per_epoch)
        multiplier = min(epoch_idx, total_iters) / total_iters
        lrs.append(start_lr + multiplier * (end_lr - start_lr))
    return lrs


def cosine_decay_lr(decay_epochs, eta_min, *, eta_max, steps_per_epoch, epochs, num_cycles=1, cycle_decay=1.0):
    """update every epoch"""
    tot_steps = steps_per_epoch * epochs
    lrs = []

    for c in range(num_cycles):
        lr_max = eta_max * (cycle_decay**c)
        delta = 0.5 * (lr_max - eta_min)
        for i in range(steps_per_epoch * decay_epochs):
            t_cur = math.floor(i / steps_per_epoch)
            t_cur = min(t_cur, decay_epochs)
            lr_cur = eta_min + delta * (1.0 + math.cos(math.pi * t_cur / decay_epochs))
            if len(lrs) < tot_steps:
                lrs.append(lr_cur)
            else:
                break

    if epochs > num_cycles * decay_epochs:
        for i in range((epochs - (num_cycles * decay_epochs)) * steps_per_epoch):
            lrs.append(eta_min)

    return lrs


def create_scheduler(
    steps_per_epoch: int,
    scheduler: str = "constant",
    lr: float = 0.01,
    min_lr: float = 1e-6,
    warmup_epochs: int = 3,
    warmup_factor: float = 0.01,
    decay_epochs: int = 10,
    num_epochs: int = 200,
    num_cycles: int = 1,
    cycle_decay: float = 1.0,
):
    r"""Creates learning rate scheduler by name.

    Args:
        steps_per_epoch: number of steps per epoch.
        scheduler: scheduler name like 'constant', 'cosine_decay', 'step_decay',
            'exponential_decay', 'polynomial_decay', 'multi_step_decay'. Default: 'constant'.
        lr: learning rate value. Default: 0.01.
        min_lr: lower lr bound for 'cosine_decay' schedulers. Default: 1e-6.
        warmup_epochs: epochs to warmup LR, if scheduler supports. Default: 3.
        warmup_factor: the warmup phase of scheduler is a linearly increasing lr,
            the beginning factor is `warmup_factor`, i.e., the lr of the first step/epoch is lr*warmup_factor,
            and the ending lr in the warmup phase is lr. Default: 0.0
        decay_epochs: for 'cosine_decay' schedulers, decay LR to min_lr in `decay_epochs`.
            For 'step_decay' scheduler, decay LR by a factor of `decay_rate` every `decay_epochs`. Default: 10.
        decay_rate: LR decay rate. Default: 0.9.
        milestones: list of epoch milestones for 'multi_step_decay' scheduler. Must be increasing. Default: None
        num_epochs: Number of total epochs. Default: 200.
        num_cycles: Number of cycles for cosine decay and cyclic. Default: 1.
        cycle_decay: Decay rate of lr max in each cosine cycle. Default: 1.0.
        lr_epoch_stair: If True, LR will be updated in the beginning of each new epoch
            and the LR will be consistent for each batch in one epoch.
            Otherwise, learning rate will be updated dynamically in each step. Default: False.
    Returns:
        Cell object for computing LR with input of current global steps
    """

    if warmup_epochs + decay_epochs > num_epochs:
        _logger.warning("warmup_epochs + decay_epochs > num_epochs. Please check and reduce decay_epochs!")

    # lr warmup phase
    warmup_lr_scheduler = []
    if warmup_epochs > 0:
        if warmup_factor == 0:
            _logger.warning(
                "The warmup factor is set to 0, lr of 0-th epoch is always zero! " "Recommend value is 0.01."
            )
        warmup_func = linear_lr
        warmup_lr_scheduler = warmup_func(
            start_factor=warmup_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
            lr=lr,
            steps_per_epoch=steps_per_epoch,
            epochs=warmup_epochs,
        )

    # lr decay phase
    main_epochs = num_epochs - warmup_epochs
    if scheduler == "cosine_decay":
        cosine_func = cosine_decay_lr
        main_lr_scheduler = cosine_func(
            decay_epochs=decay_epochs,
            eta_min=min_lr,
            eta_max=lr,
            steps_per_epoch=steps_per_epoch,
            epochs=main_epochs,
            num_cycles=num_cycles,
            cycle_decay=cycle_decay,
        )
    elif scheduler == "constant":
        main_lr_scheduler = [lr for _ in range(steps_per_epoch * main_epochs)]
    else:
        raise ValueError(f"Invalid scheduler: {scheduler}")

    # combine
    lr_scheduler = warmup_lr_scheduler + main_lr_scheduler

    return lr_scheduler
