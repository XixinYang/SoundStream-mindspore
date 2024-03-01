'''
尝试改写mindaudio中的stft，使处理的数据格式从array变成tensor
实际soundstream并未使用这一版本
'''



import math

from mindspore import ops

# Define max block sizes(256 KB)
MAX_MEM_BLOCK = 2**8 * 2**10


def _expand_to(x, ndim, axes):
    axes_tup = tuple([axes])  # type: ignore

    shape = [1] * ndim
    for i, axi in enumerate(axes_tup):
        shape[axi] = x.shape[i]
    return x.reshape(shape)


def stft(
    waveforms,
    n_fft=512,
    win_length=None,
    hop_length=None,
    window=ops.hann_window,
    center=True,
    pad_mode="reflect",
    return_complex=True,
):
    """
    Short-time Fourier transform (STFT).

    STFT represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short
    overlapping windows.

    Args:
        waveforms (np.ndarray), 1D or 2D array represent the time-serie audio signal.
        n_fft (int): Number of fft point of the STFT. It defines the frequency resolution. The number of rows in
            the STFT matrix ``D`` is ``(1 + n_fft/2)``.
            Notes:n_fft = 2 ** n, n_fft <= win_len * (sample_rate/1000)
        win_length (int): Number of frames the sliding window used to compute the STFT.
            Notes:duration (ms) = {win_length*1000}{sample_rate} If None, win_length = n_fft.
        hop_length (int): Number of frames for the hop of the sliding window used to compute the STFT. If None,
            hop_length will be set to 1/4*n_fft.
        window (str): Name of window function specified for STFT. This function should take an integer (number of
            samples) and outputs an array to be multiplied with each window before fft.
        center (bool): If True (default), the input will be padded on both sides so that the t-th frame is centered at
            time t*hop_length. Otherwise, the t-th frame begins at time t*hop_length.
        pad_mode (str): Padding mode. Options: ["center", "reflect", "constant"]. Default: "reflect".
        return_complex (bool): Whether to return complex array or a real array for the real and imaginary components.

    Returns:
        np.ndarray, STFT

    Examples:
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.spectrum as spectrum
        >>> waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> matrix = spectrum.stft(waveform)
        (257, 9)
    """
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = win_length // 4

    fft_window = window(win_length)

    # Pad the window out to n_fft size
    fft_window = _pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = _expand_to(fft_window, ndim=1 + waveforms.ndim, axes=-2)

    # Pad the time series so that frames are centered
    if center:
        if n_fft > waveforms.shape[-1]:
            raise ValueError("n_fft={} is too small for input signal of length={}".format(n_fft, waveforms.shape[-1]))

        # How many frames depend on left padding?
        start_k = math.ceil(n_fft // 2 / hop_length)

        # What's the first frame that depends on extra right-padding?
        tail_k = (waveforms.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1

        if tail_k <= start_k:
            # If tail and head overlap, then just copy-pad the signal and carry on
            start = 0
            extra = 0
            waveforms = ops.pad(waveforms, (n_fft // 2, n_fft // 2), mode=pad_mode)
        else:
            # If tail and head do not overlap, then we can implement padding on each part separately
            # and avoid a full copy-pad

            # "Middle" of the signal starts here, and does not depend on head padding
            start = start_k * hop_length - n_fft // 2

            if waveforms.ndim == 1:
                waveforms_pre = ops.pad(
                    waveforms.unsqueeze(0)[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1],
                    (n_fft // 2, 0),
                    mode=pad_mode,
                )[0]
            else:
                waveforms_pre = ops.pad(
                    waveforms[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1],
                    (n_fft // 2, 0),
                    mode=pad_mode,
                )
            af_frames = frame(waveforms_pre, frame_length=n_fft, hop_length=hop_length)[..., :start_k]
            the_shape_of_frames = af_frames.shape
            extra = the_shape_of_frames[-1]

            if tail_k * hop_length - n_fft // 2 + n_fft <= waveforms.shape[-1] + n_fft // 2:
                y_post = ops.pad(
                    waveforms[..., (tail_k) * hop_length - n_fft // 2 :],
                    (0, n_fft // 2),
                    mode=pad_mode,
                )
                y_frames_post = frame(y_post, frame_length=n_fft, hop_length=hop_length)
                extra += y_frames_post.shape[-1]
            else:
                # the end padding
                post_shape = list(the_shape_of_frames)
                post_shape[-1] = 0
                y_frames_post = ops.zeros(post_shape).to(af_frames.dtype)
    else:
        start = 0
        extra = 0
        if n_fft > waveforms.shape[-1]:
            raise ValueError(
                f"n_fft={n_fft} is too large for uncentered analysis of input signal of length={waveforms.shape[-1]}"
            )
    # Window the time series.
    y_frames = frame(waveforms[..., start:], frame_length=n_fft, hop_length=hop_length)
    shape = list(y_frames.shape)
    shape[-2] = 1 + n_fft // 2
    shape[-1] += extra

    # Fill in the warm-up
    rfft = ops.FFTWithSize(signal_ndim=1, inverse=False, real=True, onesided=True)
    off_start = af_frames.shape[-1] if center and extra > 0 else 0
    if fft_window.ndim == 2:
        stft_matrix_offstart = rfft((fft_window * af_frames).transpose()).transpose()
    if fft_window.ndim == 3:
        stft_matrix_offstart = ops.concat(
            tuple(map(lambda t: rfft(t[0].transpose()).transpose().unsqueeze(0), (fft_window * af_frames).chunk(2)))
        )

    off_end = y_frames_post.shape[-1]
    if off_end > 0:
        if fft_window.ndim == 2:
            stft_matrix_offend = rfft((fft_window * y_frames_post).transpose()).transpose()
        if fft_window.ndim == 3:
            stft_matrix_offend = ops.concat(
                tuple(
                    map(
                        lambda t: rfft(t[0].transpose()).transpose().unsqueeze(0), (fft_window * y_frames_post).chunk(2)
                    )
                )
            )

    n_columns = max(int(MAX_MEM_BLOCK // (math.prod(y_frames.shape[:-1]) * y_frames.itemsize)), 1)

    for bl_s in range(0, y_frames.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, y_frames.shape[-1])
        if fft_window.ndim == 2:
            stft_matrix_n = rfft((fft_window * y_frames[..., bl_s:bl_t]).transpose()).transpose()
        if fft_window.ndim == 3:
            stft_matrix_n = ops.concat(
                tuple(
                    map(
                        lambda t: rfft(t[0].transpose()).transpose().unsqueeze(0),
                        (fft_window * y_frames[..., bl_s:bl_t]).chunk(2),
                    )
                )
            )
        stft_matrix_offstart = ops.cat((stft_matrix_offstart, stft_matrix_n), -1)
    stft_matrix = stft_matrix_offstart if off_start else stft_matrix_offstart[..., off_start:]
    stft_matrix = ops.cat((stft_matrix, stft_matrix_offend), -1) if off_end else stft_matrix_offstart

    if return_complex:
        return stft_matrix
    else:
        return ops.view_as_real(stft_matrix)


def frame(x, frame_length=2048, hop_length=64):
    """
    Generate series of frames of the input signal.

    Args:
        x (np.ndarray, Tensor): Input audio signal.
        frame_length (int): The length as to form a group.
        hop_length (int): The hopping length.

    Returns:
        np.ndarray or Tensor, framed signals.
    """
    # x = np.array(x, copy=False)

    if hop_length < 1:
        raise ValueError("Invalid hop_length: {:d}".format(hop_length))

    num_frame = (x.shape[-1] - frame_length) // hop_length + 1
    x_frames = ops.zeros(x.shape[:-1] + (frame_length, num_frame), dtype=ms.float64)
    for i in range(frame_length):
        x_frames[..., i, :] = x[..., i : i + num_frame * hop_length][..., ::hop_length]
    return x_frames


def _pad_center(data, size, axis=-1):
    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ValueError(("Target size ({:d}) must be " "at least input size ({:d})").format(size, n))

    return ops.pad(data, lengths)


import mindspore as ms
from mindspore import ops

s = stft(ops.arange(9920, dtype=ms.float64), 1024, 256, 1024, ops.hann_window, True, "reflect", True)
print(1)
