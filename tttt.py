from soundstream import SoundStream, SoundStreamTrainer
from mindspore import ops
import mindspore as ms
import numpy as np


soundstream = SoundStream(
    codebook_size = 4096,
    rq_num_quantizers = 8,
    rq_groups = 2,                       # this paper proposes using multi-headed residual vector quantization - https://arxiv.org/abs/2305.02765
    use_lookup_free_quantizer = True,    # whether to use residual lookup free quantization - there are now reports of successful usage of this unpublished technique
    use_finite_scalar_quantizer = False, # whether to use residual finite scalar quantization
    attn_window_size = 128,              # local attention receptive field at bottleneck
    attn_depth = 2                       # 2 local attention transformer blocks - the soundstream folks were not experts with attention, so i took the liberty to add some. encodec went with lstms, but attention should be better
)

# # 生成推理结果
# audio = ops.arange(10080).to(ms.float32)
# # ms.save_checkpoint(soundstream.parameters_dict(),"D://桌面//a.ckpt")
# ms.load_checkpoint("D://桌面//a.ckpt", soundstream)
# recons = soundstream(audio, return_recons_only = True,return_discr_loss=False, return_loss_breakdown=True) # (1, 10080) - 1 channel




# 保存参数字典，用于推理精度验证
# params = soundstream.parameters_dict()
# pt_params = {}
# n = 0
# for name in params:
#     n = n + 1
#     p = params[name]
#     if name.endswith(".beta"):
#         name = name[: name.rfind(".beta")] + ".bias"
#     if name.endswith(".gamma"):
#         name = name[: name.rfind(".gamma")] + ".weight"
#     if name.endswith(".moving_mean"):
#         name = name[: name.rfind(".moving_mean")] + ".running_mean"
#     if name.endswith(".moving_variance"):
#         name = name[: name.rfind(".moving_variance")] + ".running_var"
#     if name.endswith(".embedding_table"):
#         name = name[: name.rfind(".embedding_table")] + ".weight"
#     if name[0].isdigit():
#         name = "layers." + name
#     pt_params[name] = p.value().asnumpy()
#     if "conv" in name and "weight" in name:
#         if name=="stft_discriminator.final_conv.weight":
#             pass
#         elif p.value().shape[-2]==1:
#             pt_params[name] = p.value().squeeze(-2).asnumpy()
#
#
# np.save("D://桌面//params.npy", pt_params)
# # 推理结果相对差异：
# fake_x: -1.4520709e-08 (return_recons_only = True)
# discr_loss: -3.1396890471159605e-06 (return_recons_only = False,return_discr_loss=True, return_loss_breakdown=True)
# total_loss: -2.828925003783687e-07 (return_recons_only = False,return_discr_loss=False, return_loss_breakdown=True)
# recon_loss: 0
# multi_spectral_recon_loss: -5.416478135217189e-07
# adversarial_loss: 6.5755984e-08
# feature_loss: -8.989572e-08
# all_commitment_loss: 0





# # 训练数据生成
# import mindaudio
# from mindspore import ops
# import mindspore as ms
# samplerate = 10080
# fs = 100
# amplitude = np.iinfo(np.int16).max
# n_samples=10
# for i in range(n_samples):
#     t = ops.rand((samplerate,), dtype=ms.float32).asnumpy()
#     data = amplitude * np.sin(2. * np.pi * fs * t)
#     mindaudio.write(f"D://桌面//dataset//{i}.wav", data, samplerate)
# samplerate = 20
# for i in range(n_samples):
#     t = ops.rand((samplerate,), dtype=ms.float32).asnumpy()
#     data = amplitude * np.sin(2. * np.pi * fs * t)
#     mindaudio.write(f"D://桌面//dataset//{i+10}.wav", data, samplerate)




# # 训练结果验证
trainer = SoundStreamTrainer(
    soundstream,
    folder="D://桌面//dataset",
    results_folder="D://桌面//results",
    batch_size = 2,
    grad_accum_every = 8,         # effective batch size of 32
    data_max_length_seconds = 2,  # train on 2 second audio
    num_train_steps = 10
)

trainer.train()




#### 测试stft替代的精度能否对齐
# from mindspore import ops
# from mindaudio import stft
# import librosa
# import mindspore as ms
# input=ops.arange(9920).to(ms.float32).asnumpy()
# m = stft(input, 1024, 256, 1024, "hann", True, "reflect", True)
# n = librosa.stft(input, n_fft=1024, hop_length=256, win_length=1024, window="hann",
#                  center=True, pad_mode="reflect")
# print((m - n).max())

