from copy import deepcopy

from beartype import beartype
from beartype.typing import Optional, Set

import mindspore as ms
from mindspore import Tensor, ops
from mindspore.nn import Cell


class EMA(Cell):
    """
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 2/3.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """

    @beartype
    def __init__(
        self,
        model: Cell,
        ema_model: Optional[
            Cell
        ] = None,  # if your model has lazylinears or other types of non-deepcopyable modules, you can pass in your own ema model
        beta=0.9999,
        karras_beta=False,  # if True, uses the karras time dependent beta
        update_after_step=100,
        update_every=10,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        param_names_no_ema: Set[str] = set(),
        ignore_names: Set[str] = set(),
        ignore_startswith_names: Set[str] = set(),
        include_online_model=True,  # set this to False if you do not wish for the online model to be saved along with the ema model (managed externally)
    ):
        super().__init__()
        self._beta = beta
        self.karras_beta = karras_beta

        # whether to include the online model within the module tree, so that state_dict also saves it

        self.include_online_model = include_online_model

        if include_online_model:
            self.online_model = model
        else:
            self.online_model = [model]  # hack

        # ema model

        self.ema_model = ema_model

        if self.ema_model is None:
            try:
                self.ema_model = deepcopy(model)
            except Exception as e:
                print(f"Error: While trying to deepcopy model: {e}")
                print("Your model was not copyable. Please make sure you are not using any LazyLinear")
                exit()

        self.ema_model.requires_grad = False

        # parameter names

        self.parameter_names = {
            name
            for name in self.ema_model.parameters_dict()
            if self.ema_model.parameters_dict()[name].dtype in [ms.float32, ms.float16]
        }

        # updating hyperparameters

        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_names_no_ema, (set, list))
        self.param_names_no_ema = param_names_no_ema

        self.ignore_names = ignore_names
        self.ignore_startswith_names = ignore_startswith_names

        # init and step states

        self.initted = Tensor(False)
        self.step = Tensor(0)

    @property
    def model(self):
        return self.online_model if self.include_online_model else self.online_model[0]

    @property
    def beta(self):
        if self.karras_beta:
            return (1 - 1 / (self.step + 1)) ** (1 + self.power)

        return self._beta

    def eval(self):
        return self.ema_model.set_train(False)

    def get_params_iter(self, model):
        for name, param in model.parameters_dict():
            if name not in self.parameter_names:
                continue
            yield name, param

    def copy_params_from_model_to_ema(self):
        for (_, ma_params), (_, current_params) in zip(
            self.get_params_iter(self.ema_model), self.get_params_iter(self.model)
        ):
            ops.assign(ma_params, current_params.data)

    def get_current_decay(self):
        epoch = (self.step - self.update_after_step - 1).clamp(min=0.0)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power

        if epoch.item() <= 0:
            return 0.0

        return value.clamp(min=self.min_value, max=self.beta).item()

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted = Tensor(True)

        self.update_moving_average(self.ema_model, self.model)

    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(
            self.get_params_iter(current_model), self.get_params_iter(ma_model)
        ):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_names_no_ema:
                ops.assign(ma_params, current_params.data)
                continue

            ops.assign(ma_params, ma_params.data.lerp(current_params.data, 1.0 - current_decay))

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
