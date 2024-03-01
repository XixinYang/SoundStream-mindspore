""" optim factory """
from typing import Optional

import numpy as np

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register

try:
    from mindspore import jit
except ImportError:
    from mindspore import ms_function as jit

__all__ = ["create_optimizer"]


def _check_param_value(beta1, beta2, eps, prim_name):
    """Check the type of inputs."""
    assert isinstance(beta1, float) and 0 <= beta1 <= 1.0, f"For {prim_name}, beta1 should between 0 and 1"
    assert isinstance(beta2, float) and 0 <= beta2 <= 1.0, f"For {prim_name}, beta2 should between 0 and 1"
    assert isinstance(eps, float) and eps > 0, f"For {prim_name}, eps should be bigger than 0"


_grad_scale = ops.MultitypeFuncGraph("grad_scale")
map_ = ops.Map()


@_grad_scale.register("Number", "Tensor")
def tensor_grad_scale(scale, grad):
    """Get grad with scale."""
    if scale == 1.0:
        return grad
    return ops.mul(grad, ops.cast(scale, grad.dtype))


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale_with_tensor(scale, grad):
    """Get grad with scale."""
    return ops.mul(grad, ops.cast(scale, grad.dtype))


def scale_grad(gradients, reciprocal_scale):
    gradients = map_(ops.partial(_grad_scale, reciprocal_scale), gradients)
    return gradients


_adam_opt = ops.MultitypeFuncGraph("adam_opt")
_scaler_one = Tensor(1, ms.int32)


@_adam_opt.register(
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Bool",
    "Bool",
)
def _update_run_op(
    beta1_power,
    beta2_power,
    beta1,
    beta2,
    eps,
    lr,
    weight_decay,
    param,
    m,
    v,
    gradient,
    decay_flag,
    optim_filter,
):
    """
    Update parameters.
    Args:
        beta1 (Tensor): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
        beta2 (Tensor): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
        eps (Tensor): Term added to the denominator to improve numerical stability. Should be greater than 0.
        lr (Tensor): Learning rate.
        weight_decay (Tensor): Weight decay. Should be equal to or greater than 0.
        param (Tensor): Parameters.
        m (Tensor): m value of parameters.
        v (Tensor): v value of parameters.
        gradient (Tensor): Gradient of parameters.
        decay_flag (bool): Applies weight decay or not.
        optim_filter (bool): Applies parameter update or not.
    Returns:
        Tensor, the new value of v after updating.
    """
    if optim_filter:
        param_fp32 = ops.cast(param, ms.float32)
        m_fp32 = ops.cast(m, ms.float32)
        v_fp32 = ops.cast(v, ms.float32)
        gradient_fp32 = ops.cast(gradient, ms.float32)

        next_m = ops.mul(beta1, m_fp32) + ops.mul(
            ops.cast(ops.tuple_to_array((1.0,)), ms.float32) - beta1, gradient_fp32
        )

        next_v = ops.mul(beta2, v_fp32) + ops.mul(
            ops.cast(ops.tuple_to_array((1.0,)), ms.float32) - beta2, ops.square(gradient_fp32)
        )

        regulate_m = next_m / (_scaler_one - beta1_power)
        regulate_v = next_v / (_scaler_one - beta2_power)

        update = regulate_m / (eps + ops.sqrt(regulate_v))
        if decay_flag:
            update = ops.mul(weight_decay, param_fp32) + update

        update_with_lr = ops.mul(lr, update)
        next_param = param_fp32 - ops.reshape(update_with_lr, ops.shape(param_fp32))

        next_param = ops.depend(next_param, ops.assign(param, ops.cast(next_param, param.dtype)))
        next_param = ops.depend(next_param, ops.assign(m, ops.cast(next_m, m.dtype)))
        next_param = ops.depend(next_param, ops.assign(v, ops.cast(next_v, v.dtype)))

        return ops.cast(next_param, param.dtype)
    return gradient


class AdamW(Optimizer):
    """
    Implements the gradient clipping by norm for a AdamWeightDecay optimizer.
    """

    @opt_init_args_register
    def __init__(
        self,
        params,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        loss_scale=1.0,
        clip=False,
    ):
        super().__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self.parameters.clone(prefix="adam_m", init="zeros")
        self.moments2 = self.parameters.clone(prefix="adam_v", init="zeros")
        self.hyper_map = ops.HyperMap()
        self.beta1_power = Parameter(initializer(1, [1], ms.float32), name="beta1_power")
        self.beta2_power = Parameter(initializer(1, [1], ms.float32), name="beta2_power")

        self.reciprocal_scale = Tensor(1.0 / loss_scale, ms.float32)
        self.clip = clip

    def get_lr(self):
        """
        The optimizer calls this interface to get the learning rate for the current step. User-defined optimizers based
        on :class:`mindspore.nn.Optimizer` can also call this interface before updating the parameters.

        Returns:
            float, the learning rate of current step.
        """
        lr = self.learning_rate
        if self.dynamic_lr:
            if self.is_group_lr:
                lr = ()
                for learning_rate in self.learning_rate:
                    current_dynamic_lr = learning_rate(self.global_step).reshape(())
                    lr += (current_dynamic_lr,)
            else:
                lr = self.learning_rate(self.global_step).reshape(())
        if self._is_dynamic_lr_or_weight_decay():
            self.assignadd(self.global_step, self.global_step_increase_tensor)
        return lr

    @jit
    def construct(self, gradients):
        lr = self.get_lr()
        gradients = scale_grad(gradients, self.reciprocal_scale)
        if self.clip:
            gradients = ops.clip_by_global_norm(gradients, 5.0, None)

        beta1_power = self.beta1_power * self.beta1
        self.beta1_power = beta1_power
        beta2_power = self.beta2_power * self.beta2
        self.beta2_power = beta2_power

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(
                    ops.partial(_adam_opt, beta1_power, beta2_power, self.beta1, self.beta2, self.eps),
                    lr,
                    self.weight_decay,
                    self.parameters,
                    self.moments1,
                    self.moments2,
                    gradients,
                    self.decay_flags,
                    self.optim_filter,
                )
            else:
                optim_result = self.hyper_map(
                    ops.partial(_adam_opt, beta1_power, beta2_power, self.beta1, self.beta2, self.eps, lr),
                    self.weight_decay,
                    self.parameters,
                    self.moments1,
                    self.moments2,
                    gradients,
                    self.decay_flags,
                    self.optim_filter,
                )
        else:
            optim_result = self.hyper_map(
                ops.partial(
                    _adam_opt, beta1_power, beta2_power, self.beta1, self.beta2, self.eps, lr, self.weight_decay
                ),
                self.parameters,
                self.moments1,
                self.moments2,
                gradients,
                self.decay_flags,
                self.optim_filter,
            )
        if self.use_parallel:
            self.broadcast_params(optim_result)
        return optim_result


def init_group_params(params, weight_decay):
    decay_params = []
    no_decay_params = []

    for param in params:
        if "beta" not in param.name and "gamma" not in param.name and "bias" not in param.name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params},
        {"order_params": params},
    ]


def create_optimizer(
    params,
    opt: str = "adam",
    lr: Optional[float] = 1e-3,
    weight_decay: float = 0,
    momentum: float = 0.9,
    nesterov: bool = False,
    filter_bias_and_bn: bool = True,
    loss_scale: float = 1.0,
    eps: float = 1e-10,
    **kwargs,
):
    r"""Creates optimizer by name.

    Args:
        params: network parameters. Union[list[Parameter],list[dict]], which must be the list of parameters
            or list of dicts. When the list element is a dictionary, the key of the dictionary can be
            "params", "lr", "weight_decay","grad_centralization" and "order_params".
        opt: wrapped optimizer. You could choose like 'sgd', 'nesterov', 'momentum', 'adam', 'adamw', 'lion',
            'rmsprop', 'adagrad', 'lamb'. 'adam' is the default choose for convolution-based networks.
            'adamw' is recommended for ViT-based networks. Default: 'adam'.
        lr: learning rate: float or lr scheduler. Fixed and dynamic learning rate are supported. Default: 1e-3.
        weight_decay: weight decay factor. It should be noted that weight decay can be a constant value or a Cell.
            It is a Cell only when dynamic weight decay is applied. Dynamic weight decay is similar to
            dynamic learning rate, users need to customize a weight decay schedule only with global step as input,
            and during training, the optimizer calls the instance of WeightDecaySchedule to get the weight decay value
            of current step. Default: 0.
        momentum: momentum if the optimizer supports. Default: 0.9.
        nesterov: Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients. Default: False.
        filter_bias_and_bn: whether to filter batch norm parameters and bias from weight decay.
            If True, weight decay will not apply on BN parameters and bias in Conv or Dense layers. Default: True.
        loss_scale: A floating point value for the loss scale, which must be larger than 0.0. Default: 1.0.

    Returns:
        Optimizer object
    """

    opt = opt.lower()

    if weight_decay and filter_bias_and_bn:
        params = init_group_params(params, weight_decay)

    opt_args = dict(**kwargs)
    # if lr is not None:
    #    opt_args.setdefault('lr', lr)

    # non-adaptive: SGD, momentum, and nesterov
    if opt == "sgd":
        # note: nn.Momentum may perform better if momentum > 0.
        optimizer = nn.SGD(
            params=params,
            learning_rate=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            loss_scale=loss_scale,
            **opt_args,
        )
    elif opt in ["momentum", "nesterov"]:
        optimizer = nn.Momentum(
            params=params,
            learning_rate=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            use_nesterov=nesterov,
            loss_scale=loss_scale,
        )
    # adaptive
    elif opt == "adam":
        optimizer = nn.Adam(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            use_nesterov=nesterov,
            **opt_args,
        )
    elif opt == "adamw":
        optimizer = AdamW(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            **opt_args,
        )
    elif opt == "rmsprop":
        optimizer = nn.RMSProp(
            params=params,
            learning_rate=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            epsilon=eps,
            **opt_args,
        )
    elif opt == "adagrad":
        optimizer = nn.Adagrad(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            **opt_args,
        )
    elif opt == "lamb":
        assert loss_scale == 1.0, "Loss scaler is not supported by Lamb optimizer"
        optimizer = nn.Lamb(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            **opt_args,
        )
    else:
        raise ValueError(f"Invalid optimizer: {opt}")

    return optimizer
