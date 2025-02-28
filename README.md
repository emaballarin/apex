# [NVIDIA Apex](https://github.com/NVIDIA/apex)

Personal fork with very minor modifications, for ease of install. For any reasonable purpose, have a look at [the original](https://github.com/NVIDIA/apex)!

---

Note to self:
```
git clone --recursive --recurse-submodules --single-branch --depth=1 --shallow-submodules --branch master "https://github.com/emaballarin/apex.git"
cd apex
CUDA_HOME="$CONDA_PREFIX" PATH="$CONDA_PREFIX/bin/:$PATH" MAX_JOBS=2 pip wheel -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--bnp" --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --config-settings "--build-option=--cudnn_gbn" --config-settings "--build-option=--deprecated_fused_adam" --config-settings "--build-option=--deprecated_fused_lamb" --config-settings "--build-option=--fast_bottleneck" --config-settings "--build-option=--fast_layer_norm" --config-settings "--build-option=--fast_multihead_attn" --config-settings "--build-option=--fmha" --config-settings "--build-option=--focal_loss" --config-settings "--build-option=--fused_conv_bias_relu" --config-settings "--build-option=--index_mul_2d" --config-settings "--build-option=--nccl_p2p" --config-settings "--build-option=--peer_memory" --config-settings "--build-option=--permutation_search" --config-settings "--build-option=--transducer" --config-settings "--build-option=--xentropy" .
pip install ./apex-*
```

To reduce the build time of APEX, parallel building can be enhanced via
```bash
NVCC_APPEND_FLAGS="--threads 4" pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" ./
```
When CPU cores or memory are limited, the `--parallel` option is generally preferred over `--threads`. See [pull#1882](https://github.com/NVIDIA/apex/pull/1882) for more details.

APEX also supports a Python-only build via
```bash
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./
```
A Python-only build omits:
- Fused kernels required to use `apex.optimizers.FusedAdam`.
- Fused kernels required to use `apex.normalization.FusedLayerNorm` and `apex.normalization.FusedRMSNorm`.
- Fused kernels that improve the performance and numerical stability of `apex.parallel.SyncBatchNorm`.
- Fused kernels that improve the performance of `apex.parallel.DistributedDataParallel` and `apex.amp`.
`DistributedDataParallel`, `amp`, and `SyncBatchNorm` will still be usable, but they may be slower.


### [Experimental] Windows
`pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" .` may work if you were able to build Pytorch from source
on your system. A Python-only build via `pip install -v --no-cache-dir .` is more likely to work.  
If you installed Pytorch in a Conda environment, make sure to install Apex in that same environment.


## Custom C++/CUDA Extensions and Install Options

If a requirement of a module is not met, then it will not be built.

|  Module Name  |  Install Option  |  Misc  |
|---------------|------------------|--------|
|  `apex_C`     |  `--cpp_ext`     | |
|  `amp_C`      |  `--cuda_ext`    | |
|  `syncbn`     |  `--cuda_ext`    | |
|  `fused_layer_norm_cuda`  |  `--cuda_ext`  | [`apex.normalization`](./apex/normalization) |
|  `mlp_cuda`   |  `--cuda_ext`    | |
|  `scaled_upper_triang_masked_softmax_cuda`  |  `--cuda_ext`  | |
|  `generic_scaled_masked_softmax_cuda`  |  `--cuda_ext`  | |
|  `scaled_masked_softmax_cuda`  |  `--cuda_ext`  | |
|  `fused_weight_gradient_mlp_cuda`  |  `--cuda_ext`  | Requires CUDA>=11 |
|  `permutation_search_cuda`  |  `--permutation_search`  | [`apex.contrib.sparsity`](./apex/contrib/sparsity)  |
|  `bnp`        |  `--bnp`         |  [`apex.contrib.groupbn`](./apex/contrib/groupbn) |
|  `xentropy`   |  `--xentropy`    |  [`apex.contrib.xentropy`](./apex/contrib/xentropy)  |
|  `focal_loss_cuda`  |  `--focal_loss`  |  [`apex.contrib.focal_loss`](./apex/contrib/focal_loss)  |
|  `fused_index_mul_2d`  |  `--index_mul_2d`  |  [`apex.contrib.index_mul_2d`](./apex/contrib/index_mul_2d)  |
|  `fused_adam_cuda`  |  `--deprecated_fused_adam`  |  [`apex.contrib.optimizers`](./apex/contrib/optimizers)  |
|  `fused_lamb_cuda`  |  `--deprecated_fused_lamb`  |  [`apex.contrib.optimizers`](./apex/contrib/optimizers)  |
|  `fast_layer_norm`  |  `--fast_layer_norm`  |  [`apex.contrib.layer_norm`](./apex/contrib/layer_norm). different from `fused_layer_norm` |
|  `fmhalib`    |  `--fmha`        |  [`apex.contrib.fmha`](./apex/contrib/fmha)  |
|  `fast_multihead_attn`  |  `--fast_multihead_attn`  |  [`apex.contrib.multihead_attn`](./apex/contrib/multihead_attn)  |
|  `transducer_joint_cuda`  |  `--transducer`  |  [`apex.contrib.transducer`](./apex/contrib/transducer)  |
|  `transducer_loss_cuda`   |  `--transducer`  |  [`apex.contrib.transducer`](./apex/contrib/transducer)  |
|  `cudnn_gbn_lib`  |  `--cudnn_gbn`  | Requires cuDNN>=8.5, [`apex.contrib.cudnn_gbn`](./apex/contrib/cudnn_gbn) |
|  `peer_memory_cuda`  |  `--peer_memory`  |  [`apex.contrib.peer_memory`](./apex/contrib/peer_memory)  |
|  `nccl_p2p_cuda`  |  `--nccl_p2p`  | Requires NCCL >= 2.10, [`apex.contrib.nccl_p2p`](./apex/contrib/nccl_p2p)  |
|  `fast_bottleneck`  |  `--fast_bottleneck`  |  Requires `peer_memory_cuda` and `nccl_p2p_cuda`, [`apex.contrib.bottleneck`](./apex/contrib/bottleneck) |
|  `fused_conv_bias_relu`  |  `--fused_conv_bias_relu`  | Requires cuDNN>=8.4, [`apex.contrib.conv_bias_relu`](./apex/contrib/conv_bias_relu) |
