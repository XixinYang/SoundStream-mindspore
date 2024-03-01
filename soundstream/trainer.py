import logging
import re
from functools import partial
from pathlib import Path
from shutil import rmtree

import mindaudio
from beartype import beartype
from beartype.typing import List, Optional, Type, Union
from beartype.vale import Is
from typing_extensions import Annotated

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.dataset import GeneratorDataset, config
from mindspore.experimental.optim.lr_scheduler import LambdaLR, LRScheduler

from .data import SoundDataset
from .ema import EMA
from .logger import set_logger
from .optimizer import create_optimizer
from .scheduler import create_scheduler
from .soundstream import SoundStream

# constants

logger = logging.getLogger("soundstream.train")

DEFAULT_SAMPLE_RATE = 16000

ConstantLRScheduler = partial(LambdaLR, lr_lambda=lambda step: 1.0)

# make sure only one trainer is instantiated

ONE_TRAINER_INSTANTIATED = False


def check_one_trainer():
    global ONE_TRAINER_INSTANTIATED
    assert not ONE_TRAINER_INSTANTIATED, "only one Trainer can be instantiated at a time for training"
    ONE_TRAINER_INSTANTIATED = True


# for automatically routing data emitted from a dataset to keywords of the transformer wrappers

DATASET_FIELD_TYPE_CONFIG = dict(
    raw_wave=Annotated[Tensor, Is[lambda t: t.dtype == ms.float32 and t.ndim in {2, 3}]],
    text=List[str],
    text_embeds=Annotated[Tensor, Is[lambda t: t.dtype == ms.float32 and t.ndim == 3]],
)

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def noop(*args, **kwargs):
    pass


def find_first(cond, arr):
    for el in arr:
        if cond(el):
            return el
    return None


def cycle(dl):
    while True:
        for data in dl:
            yield data


def yes_or_no(question):
    answer = input(f"{question} (y/n) ")
    return answer.lower() in ("yes", "y")


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


def checkpoint_num_step(checkpoint_path):
    """Returns the number of step trained from a checkpoint based on the filename.

    Filename format assumed to be something like "/path/to/semantic.transformer.200.pt" which is
    for 20k train step. Returns 20000 in that case.
    """
    results = re.findall(r"\d+", str(checkpoint_path))

    if len(results) == 0:
        return 0

    return int(results[-1])


@ms.jit_class
class Accumulator:
    def __init__(self, optimizer, accumulate_step):
        self.optimizer = optimizer
        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init="zeros")
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init="zeros")
        self.counter = Parameter(Tensor(1, ms.int32), "counter_")
        assert accumulate_step > 0
        self.accumulate_step = accumulate_step
        self.map = ops.HyperMap()

    def __call__(self, grads, max_grad_norm, discr=False):
        # 将单步获得的梯度累加至Accumulator的inner_grads
        self.map(ops.partial(ops.assign_add), self.inner_grads, grads)
        if self.counter % self.accumulate_step == 0:
            if exists(max_grad_norm):
                if not discr:
                    self.map(
                        ops.partial(ops.assign), self.inner_grads, ops.clip_by_norm(self.inner_grads, max_grad_norm)
                    )
                else:
                    clipped_discriminators_params = ops.clip_by_norm(
                        [param for param in self.inner_grads if param.name.startwith("discriminators")],
                        self.discr_max_grad_norm,
                    )
                    clipped_stft_discriminators_params = ops.clip_by_norm(
                        [param for param in self.inner_grads if param.name.startwith("stft_discriminator")],
                        self.discr_max_grad_norm,
                    )

                    clippped_params = {}
                    for param in clipped_discriminators_params + clipped_stft_discriminators_params:
                        clippped_params[param.name] = param

                    for param in self.inner_grads:
                        if param.name in clippped_params.keys():
                            ops.assign(param, clippped_params[param.name])

                # self.accelerator.clip_grad_norm_(self.soundstream.discriminators.parameters(), self.discr_max_grad_norm)
                # self.accelerator.clip_grad_norm_(self.soundstream.stft_discriminator.parameters(),
                #                                  self.discr_max_grad_norm)
            # 如果达到累积步数，进行参数优化更新
            self.optimizer(self.inner_grads)
            # 完成参数优化更新后，清零inner_grads
            self.map(ops.partial(ops.assign), self.inner_grads, self.zeros)
        # 计算步数加一
        ops.assign_add(self.counter, Tensor(1, ms.int32))

        return True


# main trainer class


class SoundStreamTrainer(nn.Cell):
    @beartype
    def __init__(
        self,
        soundstream: SoundStream,
        *,
        num_train_steps: int,
        batch_size: int,
        data_max_length: int = None,
        data_max_length_seconds: Union[int, float] = None,
        folder: str = None,
        lr: float = 2e-4,
        grad_accum_every: int = 4,
        wd: float = 0.0,
        warmup_steps: int = 1000,
        scheduler: Optional[Type[LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        discr_warmup_steps: Optional[int] = None,
        discr_scheduler: Optional[Type[LRScheduler]] = None,
        discr_scheduler_kwargs: dict = dict(),
        max_grad_norm: float = 0.5,
        discr_max_grad_norm: float = None,
        save_results_every: int = 5,
        save_model_every: int = 5,
        log_losses_every: int = 1,
        results_folder: str = "./results",
        valid_frac: float = 0.05,
        random_split_seed: int = 42,
        use_ema: bool = True,
        ema_beta: float = 0.995,
        ema_update_after_step: int = 500,
        ema_update_every: int = 1,
        dl_num_workers: int = 1,
        dataloader_drop_last=True,
        force_clear_prev_results: bool = None,  # set to True | False to skip the prompt
        scheduler_mode="constant",
    ):
        """
        Initialize with a SoundStream instance and either a folder containing audio data or
        train/val DataLoader instances.
        """
        super().__init__()
        check_one_trainer()

        self.soundstream = soundstream

        self.use_ema = use_ema
        if self.use_ema:
            self.ema_soundstream = EMA(
                soundstream, beta=ema_beta, update_after_step=ema_update_after_step, update_every=ema_update_every
            )

        self.step = Tensor(0)

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        hyperparameters = {
            "num_train_steps": num_train_steps,
            "batch_size": batch_size,
            "gradient_accum_every": grad_accum_every,
            "learning_rate": lr,
            "target_sample_hz": soundstream.target_sample_hz,
        }

        # create dataset

        if exists(data_max_length_seconds):
            assert not exists(data_max_length)
            data_max_length = int(data_max_length_seconds * soundstream.target_sample_hz)
        else:
            assert exists(data_max_length)

        hyperparameters["data_max_length"] = data_max_length

        dataset = SoundDataset(
            folder,
            max_length=data_max_length,
            target_sample_hz=soundstream.target_sample_hz,
            seq_len_multiple_of=soundstream.seq_len_multiple_of,
        )

        assert len(dataset) >= batch_size, "dataset must have sufficient samples for training"

        # dataloader

        dataloader = GeneratorDataset(dataset, column_names=["data"], shuffle=True, num_parallel_workers=dl_num_workers)

        # maybe split for validation

        if valid_frac > 0:
            config.set_seed(random_split_seed)
            train_dataset, val_dataset = dataloader.split([1 - valid_frac, valid_frac])
            ops.Print()(
                f"training with dataset of {len(train_dataset)*batch_size} samples and validating with randomly splitted {len(val_dataset)*batch_size} samples"
            )
        else:
            train_dataset = dataloader
            val_dataset = dataloader
            ops.Print(f"training with shared training and valid dataset of {len(train_dataset)*batch_size} samples")

        self.train_ds = train_dataset.batch(batch_size, drop_remainder=dataloader_drop_last)
        self.valid_ds = val_dataset.batch(batch_size, drop_remainder=dataloader_drop_last)

        assert exists(self.train_ds) and exists(self.valid_ds)

        trainset_size = len(train_dataset) * batch_size

        # create learning rate schedule
        if not scheduler:
            scheduler = create_scheduler(
                int(trainset_size / batch_size),
                scheduler=scheduler_mode,
                lr=lr,
                warmup_epochs=int(warmup_steps * batch_size / trainset_size),
                num_epochs=int(num_train_steps * batch_size / trainset_size),
                **scheduler_kwargs,
            )
        if not discr_scheduler:
            discr_warmup_steps = default(discr_warmup_steps, warmup_steps)
            discr_scheduler = create_scheduler(
                int(trainset_size / batch_size),
                scheduler=scheduler_mode,
                lr=lr,
                warmup_epochs=int(discr_warmup_steps * batch_size / trainset_size),
                num_epochs=int(num_train_steps * batch_size / trainset_size),
                **discr_scheduler_kwargs,
            )

        # optimizers

        self.optim = create_optimizer(
            soundstream.non_discr_parameters(),
            opt="adamw",  # free to choose sgd/momentum/adam/adamw/rmsprop/adagrad/lamb
            lr=scheduler,
            weight_decay=wd,
        )

        for discr_optimizer_key, discr in self.multiscale_discriminator_iter():
            one_multiscale_discr_optimizer = create_optimizer(
                discr.trainable_params(),
                opt="adamw",  # free to choose sgd/momentum/adam/adamw/rmsprop/adagrad/lamb
                lr=discr_scheduler,
                weight_decay=wd,
            )
            setattr(self, discr_optimizer_key, one_multiscale_discr_optimizer)

        self.discr_optim = create_optimizer(
            soundstream.stft_discriminator.trainable_params(),
            opt="adamw",  # free to choose sgd/momentum/adam/adamw/rmsprop/adagrad/lamb
            lr=discr_scheduler,
            weight_decay=wd,
        )

        self.accumulator = Accumulator(self.optim, self.grad_accum_every)

        # max grad norm
        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm

        # prepare the multiscale discriminators

        for name, _ in self.multiscale_discriminator_iter():
            optimizer = getattr(self, name)
            setattr(self, name, optimizer)

        # dataloader iterators

        self.train_ds_iter = cycle(self.train_ds)
        self.valid_ds_iter = cycle(self.valid_ds)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.log_losses_every = log_losses_every

        self.results_folder = Path(results_folder)
        self.log_path = results_folder

        if force_clear_prev_results is True or (
            not exists(force_clear_prev_results)
            and len([*self.results_folder.glob("**/*")]) > 0
            and yes_or_no("do you want to clear previous experiment checkpoints and results?")
        ):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents=True, exist_ok=True)

        # save tracker hyperparameters

        self.tracker_hps = hyperparameters

    @property
    def ema_tokenizer(self):
        return self.ema_soundstream.ema_model

    def tokenize(self, audio):
        return self.ema_tokenizer.tokenize(audio)

    def set_model_as_ema_model_(self):
        """this will force the main 'online' model to have same parameters as the exponentially moving averaged model"""
        assert self.use_ema
        ms.load_param_into_net(self.ema_soundstream.ema_model, self.soundstream.parameters_dict())

    def save(self, path):
        pkg = dict(
            model=self.soundstream.parameters_dict(),
            optim=self.optim.parameters_dict(),
            config=self.soundstream._configs,
            discr_optim=self.discr_optim.parameters_dict(),
        )

        if self.use_ema:
            pkg["ema_model"] = self.ema_soundstream.parameters_dict()

        for key, _ in self.multiscale_discriminator_iter():
            discr_optim = getattr(self, key)
            pkg[key] = discr_optim.parameters_dict()

        ms.save_checkpoint(pkg, path)

    def multiscale_discriminator_iter(self):
        for ind, discr in enumerate(self.soundstream.discriminators):
            yield f"multiscale_discr_optimizer_{ind}", discr

    def multiscale_discriminator_optim_iter(self):
        for name, _ in self.multiscale_discriminator_iter():
            yield name, getattr(self, name)

    def train_step(self):
        step = int(self.step.item())
        log_losses = self.log_losses_every > 0 and not (step % self.log_losses_every)

        self.soundstream.set_train()

        # logs

        logs = {}

        # update vae (generator)

        (wave,) = next(self.train_ds_iter)

        grad_fn = ms.value_and_grad(
            partial(self.soundstream, return_loss_breakdown=True),
            None,
            self.soundstream.trainable_params(),
            has_aux=True,
        )
        (
            loss,
            (recon_loss, multi_spectral_recon_loss, adversarial_loss, feature_loss, all_commitment_loss),
            grads,
        ) = grad_fn(wave, wave)
        loss = ops.depend(
            loss / self.grad_accum_every, self.accumulator(grads, self.max_grad_norm)
        )

        accum_log(
            logs,
            dict(
                loss=loss.item() / self.grad_accum_every,
                recon_loss=recon_loss.item() / self.grad_accum_every,
            ),
        )

        if log_losses:
            accum_log(
                logs,
                dict(
                    multi_spectral_recon_loss=multi_spectral_recon_loss.item() / self.grad_accum_every,
                    adversarial_loss=adversarial_loss.item() / self.grad_accum_every,
                    feature_loss=feature_loss.item() / self.grad_accum_every,
                    all_commitment_loss=all_commitment_loss.item() / self.grad_accum_every,
                ),
            )

        # update discriminator
        (wave,) = next(self.train_ds_iter)

        grad_fn = ms.value_and_grad(
            partial(self.soundstream, return_discr_loss=True, return_discr_losses_separately=True),
            None,
            self.soundstream.trainable_params(),
        )
        discr_losses, grads = grad_fn(wave, wave)

        for name, discr_loss in discr_losses:
            discr_loss = ops.depend(
                discr_loss / self.grad_accum_every, self.accumulator(grads, self.discr_max_grad_norm, discr=True)
            )
            accum_log(logs, {name: discr_loss.item() / self.grad_accum_every})

        # build pretty printed losses

        losses_str = (
            f"{step}: soundstream total loss: {logs['loss']:.3f}, soundstream recon loss: {logs['recon_loss']:.3f}"
        )

        if log_losses:
            logger.info(**logs)

        for key, loss in logs.items():
            if not key.startswith("scale:"):
                continue
            _, scale_factor = key.split(":")

            losses_str += f" | discr (scale {scale_factor}) loss: {loss:.3f}"

            if log_losses:
                logger.info(**{f"discr_loss (scale {scale_factor})": loss})

        # log

        ops.Print(losses_str)

        if self.use_ema:
            self.ema_soundstream.update()

        if not (step % self.save_results_every):
            models = [(self.soundstream, str(step))]
            if self.use_ema:
                models.append((self.ema_soundstream.ema_model if self.use_ema else self.soundstream, f"{step}.ema"))

            (wave,) = next(self.valid_ds_iter)

            for model, label in models:
                model.set_train(False)

                recons = model(wave, return_recons_only=True)

                for ind, recon in enumerate(recons.unbind(dim=0)):
                    filename = str(self.results_folder / f"sample_{label}.flac")
                    mindaudio.write(filename, recon.asnumpy(), self.soundstream.target_sample_hz)

            self.print(f"{step}: saving to {str(self.results_folder)}")

        if not (step % self.save_model_every):
            model_path = str(self.results_folder / f"soundstream.{step}.ckpt")
            self.save(model_path)

            self.print(f"{step}: saving model to {str(self.results_folder)}")

        self.step.add_(1)
        return logs

    def train(self, log_fn=noop):
        set_logger(name="soundstream", output_dir=self.log_path)
        logger.info("hyper-parameters:\n" + str(self.tracker_hps))
        while self.step < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print("training complete")
