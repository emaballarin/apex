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
