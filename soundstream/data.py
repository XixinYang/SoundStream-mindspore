from functools import wraps
from pathlib import Path

import mindaudio
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Tuple, Union

from mindspore import Tensor, ops
from mindspore.dataset import GeneratorDataset

from .utils import curtail_to_multiple

# helper functions


def exists(val):
    return val is not None


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def is_unique(arr):
    return len(set(arr)) == len(arr)


# dataset functions


class SoundDataset:
    @beartype
    def __init__(
        self,
        folder,
        target_sample_hz: Union[
            int, Tuple[int, ...]
        ],  # target sample hz must be specified, or a tuple of them if one wants to return multiple resampled
        max_length: Optional[
            int
        ] = None,  # max length would apply to the highest target_sample_hz, if there are multiple
        seq_len_multiple_of: Optional[Union[int, Tuple[Optional[int], ...]]] = None,
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), f'folder "{str(path)}" does not exist'

        files = [file for file in path.glob(f"**/*.wav")]
        assert len(files) > 0, "no sound files found"

        self.files = files

        self.max_length = max_length
        self.target_sample_hz = cast_tuple(target_sample_hz)
        num_outputs = len(self.target_sample_hz)

        # strategy, if there are multiple target sample hz, would be to resample to the highest one first
        # apply the max lengths, and then resample to all the others

        self.max_target_sample_hz = max(self.target_sample_hz)
        self.seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)

        assert len(self.target_sample_hz) == len(self.seq_len_multiple_of)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        data, sample_hz = mindaudio.read(file)
        data = Tensor(data)
        assert data.numel() > 0, f"one of your audio file ({file}) is empty. please remove it from your folder"

        if len(data.shape) > 1:
            # the audio has more than 1 channel, convert to mono
            data = data.transpose().mean(0, True)

        # first resample data to the max target freq

        data = Tensor(mindaudio.resample(data.asnumpy(), sample_hz, self.max_target_sample_hz, "minddata")).unsqueeze(0)
        sample_hz = self.max_target_sample_hz

        # then curtail or pad the audio depending on the max length
        max_length = self.max_length
        audio_length = data.shape[1]

        if exists(max_length):
            if audio_length > max_length:
                max_start = audio_length - max_length
                start = ops.randint(0, max_start, (1,))
                data = data[:, start : start + max_length]
            else:
                data = ops.pad(data, (0, max_length - audio_length), "constant")

        data = data.squeeze(0)

        # resample if target_sample_hz is not None in the tuple

        num_outputs = len(self.target_sample_hz)
        data = cast_tuple(data, num_outputs)

        data_tuple = tuple(
            mindaudio.resample(d, sample_hz, target_sample_hz, "minddata")
            for d, target_sample_hz in zip(data, self.target_sample_hz)
        )

        output = []

        # process each of the data resample at different frequencies individually for curtailing to multiple

        for data, seq_len_multiple_of in zip(data_tuple, self.seq_len_multiple_of):
            if exists(seq_len_multiple_of):
                data = curtail_to_multiple(data, seq_len_multiple_of)

            output.append(data.float())

        # cast from list to tuple

        output = tuple(output)

        # return only one audio, if only one target resample freq

        if num_outputs == 1:
            return output[0]

        return output


# # dataloader functions
#
# def collate_one_or_multiple_tensors(fn):
#     @wraps(fn)
#     def inner(data):
#         is_one_data = not isinstance(data[0], tuple)
#
#         if is_one_data:
#             data = fn(data)
#             return (data,)
#
#         outputs = []
#         for datum in zip(*data):
#             if is_bearable(datum, Tuple[str, ...]):
#                 output = list(datum)
#             else:
#                 output = fn(datum)
#
#             outputs.append(output)
#
#         return tuple(outputs)
#
#     return inner

# @collate_one_or_multiple_tensors
# def curtail_to_shortest_collate(data):
#     min_len = min(*[datum.shape[0] for datum in data])
#     data = [datum[:min_len] for datum in data]
#     return ops.stack(data)
#
# @collate_one_or_multiple_tensors
# def pad_to_longest_fn(data):
#     max_length=max([i.shape[0] for i in data])
#     padded_data=[ops.pad(i.T,(0,max_length-i.shape[0])).T.unsqueeze(0) for i in data]
#     return ops.stack(padded_data, 0)
