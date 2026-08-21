# DeepSeek-V4 DSpark 上机手册：A3 单机穿刺 → A2(tige) 多机

- 日期：2026-08-18
- 分支：**`dsv4-dspark-moe`**（主路径，sglang 0.5.14）；
  `dsv4-dspark-moe-0516` 是 0.5.16 环境的备选，见 §2.2
- 适用范围：**离线**链路（`prepare_hidden_states.py` 采特征 → `specforge train` 微调）。
  online / disaggregated EP 本轮明确禁止，见 0812 开发文档第 10 节第 3 条。
- 当前状态：**Stage 0 未通过**。A3 上 prepare 卡在 SGLang 的 NPU 自定义算子缺失，
  尚未产出过任何特征文件。后面所有阶段都还没有真机数据。

本文档按"前一项不通过就不要做下一项"的顺序组织。每个阶段给出：做什么、
怎么判断通过、不通过时往哪查。

---

## 0. 先读这一节：三个已修复项和一个未修复项

这轮排查在 A3 单机上依次撞了四个问题，前三个已在本分支修掉，第四个是环境阻断。

| # | 现象 | 根因 | 状态 |
| --- | --- | --- | --- |
| 1 | `RuntimeError: target model does not expose SGLang capture hook 'set_dspark_layers_to_capture'` | stock sglang 0.5.14 给 V2/V3 接了 aux capture，没给 V4 接 | 已修：`patches/sglang/v0.5.14/apply_deepseek_v4_capture.py`（925bbf2） |
| 2 | `RuntimeError: Prefill out of memory. Try to allocate 2048 tokens. Available tokens: 128` | V4 是 hybrid SWA，SWA 子池只拿 full 池的 10%；离线把 full 池设成"恰好一个 batch"，SWA 子池就只剩 128 token | 已修：`capture.py` 自动放大 + `--sglang-max-total-tokens`（5738ad1） |
| 3 | 权重/显存怀疑 | **不是问题**。36.99 GB/rank × 8 = 296 GB，与盘上 294 GB 吻合，分片和量化都正确，卡上还空 23.8 GB | 无需处理 |
| 4 | `AttributeError: '_OpNamespace' 'custom' object has no attribute 'npu_sparse_attn_sharedkv_metadata'` | 容器缺 `custom_ops`（`sgl-kernel-npu` 的 release 产物，不在 SGLang 仓库里） | **唯一未解决项**。把算子装进现有 0.5.14 容器，见 §1.3 / 附录 A |

第 4 项是**现在唯一挡路的东西**。前三项修完之后，0.5.14 上的运行已经走到：
权重加载完成（36.99 GB/rank）→ 分布式初始化完成 → KV 池建立 → capture hook 设置成功
→ `prepare_for_extend` 通过 → 进入 `model_runner.forward()` → 调 kernel 时缺件。

**也就是说 SpecForge 离线 backend 的整条链路在 0.5.14 上是真机验证过的**，
只差算子。这一点决定了下面的路线选择。

### 路线选择：留在 0.5.14

| | 0.5.14（主路径） | 0.5.16 镜像（备选） |
| --- | --- | --- |
| SpecForge 离线 backend | **真机跑到 forward 内部，已验证** | 静态核对，一行没实跑 |
| SpecForge 官方 pin | `pyproject.toml` 就是 `sglang==0.5.14` | 偏离 pin |
| 历史适配 | 全部在 0.5.14 上做的 | 需要重新建立信心 |
| V4 capture hook | 已由本分支 applier 加上（§2.4） | 上游自带 |
| 缺什么 | **只缺 `custom_ops`** | 无（镜像自带） |

有 0.5.16 镜像**不构成**切版本的理由：镜像只是 kernel 的一种交付方式。
`sgl-project/sglang#29794` 里维护者那句 "update to v0.5.16 image" 是对一个跑 server
且缺 kernel 的人说的，解决的是缺件，不是版本要求。

拿真机验证过的 0.5.14 去换未验证的 0.5.16 移植，只为绕开一个**已经知道来源**的缺件，
不划算。`dsv4-dspark-moe-0516` 分支保留为备选：0.5.14 容器装不上算子时才走。

问题 2 的修复只抬天花板不抬地板（`ModelRunner` 仍取 `min(profiled, requested)`），
普通均匀 KV 池的目标完全不受影响。子池公式已用 V4 自己报的两组数字反验：
`full=1198592 → swa=119808`、`full=2048 → swa=128`，都等于
`floor(full × ratio / page) × page`。

### 这轮排查里被证伪的方向，不要重走

- **不是显存不够。** 权重 36.99 GB/rank，卡上剩 23.8 GB 没动过。
- **`--sglang-mem-fraction-static` 在离线链路里基本是空转的。** KV 池被
  `max_total_tokens` 卡死，调这个值物理占用不变；调太低反而会让
  `profile_max_num_token` 变负，抛出一句误导性的 "Please try to increase
  --mem-fraction-static"。
- **调小 `--max-length` 是反效果。** 池子大小 = `batch × max_length`，
  请求也同步变小，跑通了也不说明问题。
- **`--sglang-ep-size` 与问题 2 无关**（那是 MoE 激活工作区，问题 2 连 forward
  都没进）。它作为 MoE 吞吐/显存调优项仍然有效，但要等 forward 真跑起来之后再评估。
- **`SGLANG_OPT_FP8_WO_A_GEMM=0` 不会导致反量化，而且是 W8A8 必需项。**
  权重数字先否掉了反量化的猜测；#29794 给出了确切理由——不设会在
  `MQALayer.__init__` 断言失败，因为 modelslim W8A8 权重没有 blockwise-FP8 scale。

---

## 1. Stage 0：环境闸门（`custom_ops`）

`torch.ops.custom.*` 是 V4 在 NPU 上的实现本体，不是可选优化：

- attention：`ascend_dsv4_backend.py::_kernel_metadata_from_parts` →
  `npu_sparse_attn_sharedkv_metadata`
- mHC：`deepseek_v4.py::hc_post` 开头就是 `if _is_npu: return
  torch.ops.custom.npu_hc_post(...)`，`hc_pre` 同理

`hc_post` 里确实有纯 torch 实现（`hc_post_torch_impl`），但被 `_is_npu` 分支短路；
attention 那条没有 fallback。**没有 A2/A3 通用、且不依赖 `custom_ops` 的替代算子**：
V4 的 CSA/HCA 稀疏注意力是模型语义的一部分，换成 dense 数值就不对了。已检索
vllm-ascend / torch_npu 公开算子，无对应物。

### 1.1 判定

```bash
python -c "import custom_ops; import torch; print(sorted(dir(torch.ops.custom))[:40])"
```

必须能看到 `npu_hc_pre` / `npu_hc_post` / `npu_sparse_attn_sharedkv_metadata`。
启动日志里如果出现下面这行，就是没装：

```
WARNING:sglang.srt.hardware_backend.npu.utils:NPU custom kernel packages unavailable: No module named 'custom_ops'
```

### 1.2 已排查的容器

| 容器 | `custom_ops` | 备注 |
| --- | --- | --- |
| a3 sglang | 无 | `sgl_kernel_npu 2026.5.1`、`triton_ascend 3.2.1`、sglang 0.5.14 源码树 |
| a2 sglang | 无 | 同上 |
| a2 vllm | 无 | `vllm_ascend 0.19.1rc2` |

`sgl_kernel_npu` 装了但不提供 `torch.ops.custom`，两者不是一回事。

### 1.3 获取途径（优先：装进现有 0.5.14 容器）

**首选：把算子装进你已有的 0.5.14 容器，不换镜像。**

那个容器的 sglang 本身就是 NPU DSV4 fork（它有 `ascend_dsv4_backend.py`、
`pool_configurator.py` 的 DSV4 pool 逻辑，stock 0.5.14 没有这些），
**它是按"运行时会有一份 kernel 包"来构建的**。所以第一步是找出它期望哪一版：

```bash
grep -nE "SGLANG_KERNEL_NPU_TAG|CANN_VERSION|DEVICE_TYPE|sgl-kernel-npu" \
    /sgl-workspace/sglang/docker/npu.Dockerfile
```

拿到 tag / CANN / DEVICE_TYPE，附录 A 里的版本猜测就全部消除，直接下对应 release。
安装步骤见[附录 A](#附录-a把-npu-算子搬进非官方容器)。

**备选：换官方镜像。** 只在 0.5.14 容器装不上算子时走这条——它会把 sglang 带到
0.5.16，SpecForge 侧要换 `dsv4-dspark-moe-0516` 分支，而那套移植没有真机验证。

SGLang 官方给 V4 出了 A3 和 910B 两个 NPU 镜像
（[sgl-project/sglang#23598](https://github.com/sgl-project/sglang/issues/23598)），
华为云 SWR 上 CANN 版本钉死的 **v0.5.16** 版本：

```
swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.0.0-a3-v0.5.16     # Stage 1，A3 单机
swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.0.0-910b-v0.5.16   # Stage 2，A2/tige
```

**910b 有独立镜像，说明 A2 不是不支持**，之前"这个算子只有 A3 能用"的判断不成立。

> 注意：我没有实际验证过这两个镜像里带 `custom_ops`，这是从"官方为该平台出了 V4
> 镜像 + 这些 op 是跑 V4 的必需品"推出来的。拉下来第一件事就是跑 §1.1。

v0.5.16 这个版本要求来自 [#29794](https://github.com/sgl-project/sglang/issues/29794)
里维护者 @randgun 2026-08-11 的回复，原文就是 "Please update to v0.5.16 image" 加上面
两个 SWR 路径。该 issue 报的正是 910B + V4-Flash-W8A8 + `npu_hc_pre` 缺失，和我们
Stage 0 撞的是同一类问题（我们先挂在 attention，他先挂在 mHC）。issue 仍 open。

### 1.4 `custom_ops` 的来源（来自容器树 `docker/npu.Dockerfile`，一手）

容器里的 SGLang 树带着构建自己的 Dockerfile，`custom_ops` 的出处写得很清楚：
它不在 SGLang 仓库里，而是 **`sgl-project/sgl-kernel-npu` 的 release 产物**，
按 `(TAG, CANN 版本, DEVICE_TYPE, arch)` 分别构建。

```
https://github.com/sgl-project/sgl-kernel-npu/releases/download/${TAG}/
    custom-ops-${TAG}-torch2.10.0-cann${CANN}-${DEVICE}-${arch}.zip
    ops-transformer-${TAG}-torch2.10.0-cann${CANN}-${DEVICE}-${arch}.zip
    sgl-kernel-npu-${TAG}-torch2.10.0-py311-cann${CANN}-${DEVICE}-${arch}.zip
```

镜像里的安装步骤（`docker/npu.Dockerfile`，`CANN=9.0.0`、`DEVICE` 取 `a3` 或 `910b`）：

```bash
unzip custom-ops-*.zip && unzip ops-transformer-*.zip && chmod +x *.run
./CANN-custom_ops-none-linux.$(arch).run        --install-path=/usr/local/Ascend/cann-9.0.0/opp
./cann-ops-transformer-custom_linux-$(arch).run --install-path=/usr/local/Ascend/cann-9.0.0/opp
pip install custom_ops-1.0-cp311-cp311-linux_$(arch).whl
# DeepEP / sgl_kernel_npu 另一个 zip
unzip sgl-kernel-npu-*.zip && pip install deep_ep*.whl sgl_kernel_npu*.whl
```

三件事因此说得通了：

1. **那两个 `.run` 装出来的就是 #29794 里要 source 的 vendor 包**——
   `opp/vendors/customize`（CANN-custom_ops）和 `opp/vendors/custom_transformer`
   （cann-ops-transformer）。CANN 侧算子和 Python 侧 `torch.ops.custom` 绑定是**两件事**，
   `.run` 装前者、`.whl` 装后者，缺一不可。这解释了为什么 #29794 的报告者 source 了
   vendor 脚本仍然缺 `npu_hc_pre`——他缺的是 whl 那半。
2. **`DEVICE_TYPE` 是构建参数，a3 和 910b 各有一套 kernel**，A2 在算子层面确实被支持。
3. 加载入口是 `hardware_backend/npu/utils.py::init_npu_backend()`，
   里面 `import custom_ops` / `import sgl_kernel_npu` 包在 try/except 里，
   所以缺了只 warning 不报错，一路拖到 forward 才炸。

**这意味着 Stage 0 不是只能靠换镜像**——理论上可以把对应 `DEVICE_TYPE` 的这几个包
装进任何 CANN 9.0.0 容器。但基础镜像也要对得上
（`quay.io/ascend/cann:9.0.0-${DEVICE}-ubuntu22.04-py3.11`、torch 2.10.0、py3.11），
**优先仍然是直接用官方镜像**，自行安装只作为镜像拉不动时的退路。具体做法见
[附录 A](#附录-a把-npu-算子搬进非官方容器)。

### 1.5 W8A8 环境配方（来自 #29794 issue body，一手）

**这一整段是跑 W8A8 权重的前提，不是调优项，而且要整段用、不要挑几条。**

这些 `SGLANG_OPT_*=False` / `=0` 里大部分的作用是**关掉 GPU-only 代码路径**。
少设一条就会在 forward 深处撞一次 `Could not find CUDA installation` 或
`Cannot detect CUDA architecture`，而这些分支散布在 attention、mHC、MoE 路由、
GEMM 各处，一条条试要试很多轮。已实测的一例：`SGLANG_OPT_USE_FUSED_HASH_TOPK`
默认 `True`（`environ.py`），会让 DSV4 的 MoE 路由走 tvm_ffi JIT 编译，
而它的 ninja 生成器无条件调 `_find_cuda_home()`。

```bash
# --- Ascend toolkit：后两行是 CANN 自定义算子 vendor 包 ---
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

# --- NPU runtime ---
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export INF_NAN_MODE_FORCE_DISABLE=1      # W8A8 必需：不设会溢出成 NaN，且不报错

# --- DSV4 ---
export IS_DEEPSEEK_V4=1
export USE_FUSED_HC_PRE_ASCENDC=1
export SGLANG_DSV4_NPU_FUSED_COMPRESSOR=1
export SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL=0

# --- 跳过 GPU-only 分支（modelslim W8A8 权重没有 blockwise-FP8 scale）---
export SGLANG_OPT_FP8_WO_A_GEMM=0        # 不设会在 MQALayer.__init__ 断言失败
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
```

两点说明：

- **`INF_NAN_MODE_FORCE_DISABLE=1` 对采特征尤其致命。** 溢出成 NaN 不会抛异常，
  会一路写进特征文件，训练时才以别的形式暴露。采完必须查 NaN，见 §2.6。
- **两个 `opp/vendors/*/bin/set_env.bash` 是 CANN 自定义算子 vendor 包。**
  报告者 source 了它们仍然缺 `npu_hc_pre`，所以这是必要不充分条件——kernel 本体
  还是要靠 v0.5.16 镜像。

### 1.6 不通过怎么办


不要尝试把 attention + mHC + compressor 三块全改回 torch 路径：工作量大、
数值无从验证、性能不可用。正确做法是拿到 kernel 包，或在 #29794 追问来源。

---

## 2. Stage 1：A3 单机穿刺（当前目标）

拓扑：单机 8 卡，TP=8 采特征，EP=8 / DP=1 训练。

### 2.1 容器与代码

**主路径：现有 0.5.14 容器 + `dsv4-dspark-moe` 分支。**

```bash
# 0.5.14 容器内。算子按 §1.3 / 附录 A 装好后：
python -c "import custom_ops; import torch; print(len(dir(torch.ops.custom)))"   # Stage 0 闸门
git clone -b dsv4-dspark-moe https://github.com/fg11991/specforge.git /sgl-workspace/SpecForge
cd /sgl-workspace/SpecForge && pip install -e . --no-deps
```

这条路上 SpecForge 的离线 backend 已经真机跑到 forward 内部，只需要按 §2.4 打
V4 capture applier。§2.2 和 §2.3 是 0.5.16 备选路径才需要看的。

**备选路径：官方 0.5.16 镜像 + `dsv4-dspark-moe-0516` 分支。**

```bash
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.0.0-a3-v0.5.16
python -c "import custom_ops; import torch; print(len(dir(torch.ops.custom)))"
git clone -b dsv4-dspark-moe-0516 https://github.com/fg11991/specforge.git /sgl-workspace/SpecForge
cd /sgl-workspace/SpecForge && pip install -e . --no-deps
```

走这条时**分支必须是 `-0516`**，离线 backend 的 0.5.16 适配只在那个分支上（§2.2），
而且那套适配没有真机验证。

**两条路 `--no-deps` 都不能省。** `pyproject.toml` 钉的是 `sglang==0.5.14`；
在 0.5.16 镜像里不带 `--no-deps`，pip 会把 0.5.16 卸掉重装，连带 `custom_ops`
依赖的那套一起废掉。在 0.5.14 容器里它同样会重装 sglang，冲掉你打好的 applier。

### 2.2 0.5.16 备选路径：不需要给 SGLang 打 patch

**上游 v0.5.16 的 `deepseek_v4.py` 已经自带 `set_dspark_layers_to_capture`**，
而且实现与本仓库 applier 的四处改动逐行等价：fused mHC 下补 `hc_post` 还原真实层
输出、`.mean(dim=1)` 折叠 mHC 流、层号不加 +1 offset、capture 时把
`hidden_states_before_norm` 置 None（那一行连写法都相同）。上游还多两道护栏：
capture 时跳过 TBO，capture + prefill CP 直接 `NotImplementedError`。

所以走 0.5.16 备选路径时 **不要跑 `apply_deepseek_v4_capture.py`**；它的 `--check`
会因锚点不匹配而失败，那是正确行为。§2.4 是 0.5.14 主路径用的。

> 顺带：`.mean(dim=1)` 这个折叠方式上游独立做了同样的选择。这不构成证明，但上游那条
> 路是给 DSpark static-verify 用的，两个独立实现一致让 §2.7 的风险明显下降。数值抽检
> 仍建议做，但不再是拦路虎级别。

要改的是 **SpecForge 自己**——它重实现了 SGLang 的分布式内部机制，0.5.16 动了四处：

| 断点 | 0.5.16 的变化 | `-0516` 分支的处理 |
| --- | --- | --- |
| `patch.py` import | `compute_dp_attention_local_info` 被删，模块 import 就炸 | 上游 `initialize_dp_attention` 签名与我们那份完全一致且不建进程组，直接委托 |
| `model_runner.py` | `get_attention_tp_group` 被删 | 改用 `runtime_context.get_parallel().attn_tp_group` |
| `capture.py` 构造 runner | `ModelRunner` 改收单个冻结的 `ParallelState`，不再收 flat 的 tp/moe-ep/pp 参数 | 按 `Scheduler` 同样的方式用 `compute_dp_attention_world_info` 推导 attention rank 后构造 |
| `init_torch_distributed` | 不再返回 pre-load 显存，改由 `alloc_memory_pool` 读 `self.pre_model_load_memory` | 同一个 `get_available_gpu_memory` 测量值，同时发布属性并保留返回值 |

四处都用能力探测（`try: import` / `getattr`）而不是版本号判断，所以 0.5.14 路径保留。
`utils.py` 完全没动——两个私有 `LogitsProcessor` 方法签名一字未变。
`ServerArgs` 的 `swa_full_tokens_ratio` / `max_total_tokens` / `page_size` /
`quantization` 都还在，**KV 池修复和 `--sglang-quantization` 在 0.5.16 上照常生效**。

**核对基准是容器里的真实代码**，不是上游 tag：
[`fg11991/sglang@dspark-dev`](https://github.com/fg11991/sglang/tree/dspark-dev)
（镜像树的快照）。逐项结果：

- `specforge/offline_capture` 下 **38 个非 try 保护的 sglang import 全部解析通过**
- `ParallelState` 18 个字段、`TorchDistributedResult` 4 个字段、
  `get_parallel().attn_tp_group` 的实现，与适配假设一致
- 两个私有 `LogitsProcessor` 方法**参数名和顺序一字未变**（只有类型标注从
  `Optional[List[torch.Tensor]]` 变成 `Optional[AuxHiddenStates]`，而后者是
  `Union[torch.Tensor, List[torch.Tensor]]` 的别名），`utils.py` 的位置参数调用不受影响
- aux 拼接从 `torch.cat` 换成了 `pack_aux_hidden_states()`，但它对 list 仍然走
  `torch.cat(..., dim=-1)`，而 V4 append 的就是普通 list，**12288 的宽度不变**
- `pool_configurator.py` 的 `align_page_size(int(full_tokens * swa_full_tokens_ratio))`
  与 KV 池修复反推出的公式完全一致
- V4 capture 实现（`completed.mean(dim=1)`、fused 补 `hc_post`、
  `hidden_states_before_norm` 置 None）与上游 v0.5.16 一致

**仍未在硬件上跑过**：本地没有 NPU 也没有 sglang，全部是静态核对。残留风险是
0.5.16 新增的 decode-context-parallel 组，我们的 `initialize_model_parallel` 重实现
没有创建它；`--dcp-size` 保持默认 1 时应该无影响。

### 2.3 起跑前的 seam 自检（0.5.16 路径）

跑重活之前先过一遍，一条命令：

```bash
cd /sgl-workspace/SpecForge && python - <<'PY'
import sglang, inspect, torch
print("sglang", sglang.__version__, sglang.__file__)

def probe(name, fn):
    try:
        print(f"[ok]   {name}: {fn()}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")

probe("custom_ops", lambda: (__import__("custom_ops"), len(dir(torch.ops.custom)))[1])
probe("backend imports", lambda: __import__(
    "specforge.offline_capture.sglang_backend.capture", fromlist=["x"]).__name__)
probe("V4 capture hooks", lambda: [
    a for a in dir(__import__("sglang.srt.models.deepseek_v4", fromlist=["x"]).DeepseekV4ForCausalLM)
    if "layers_to_capture" in a])

from sglang.srt.layers.logits_processor import LogitsProcessor
for m in ("_get_pruned_states", "_get_hidden_states_to_store"):
    probe(f"LogitsProcessor.{m}",
          lambda m=m: list(inspect.signature(getattr(LogitsProcessor, m)).parameters))

from sglang.srt.server_args import ServerArgs
probe("ServerArgs fields", lambda: [
    f for f in ("max_total_tokens", "swa_full_tokens_ratio", "page_size", "ep_size")
    if hasattr(ServerArgs, f) or f in getattr(ServerArgs, "__annotations__", {})])
PY

python -m pytest tests/test_runtime/test_sglang_0514_compat.py tests/test_offline_capture/ -q
```

判读：

| 结果 | 含义 |
| --- | --- |
| `custom_ops` FAIL | 镜像不对或 CANN 环境没 source，Stage 0 未过，停在这里 |
| `backend imports` FAIL | 0.5.16 挪了模块路径，`capture.py` 的 import 要改 |
| `V4 capture hooks` 非空 | 0.5.16 的预期结果，**跳过 §2.4 的 applier** |
| `V4 capture hooks` 为 `[]` | 只应发生在 0.5.14 上，需要打 applier，见 §2.4 |
| `LogitsProcessor._get_*` FAIL 或参数表变了 | `utils.py` 的私有方法调用要跟着改 |
| `ServerArgs fields` 缺 `swa_full_tokens_ratio` | KV 池自动放大会静默失效（有 `getattr` 兜底不会崩，但也不生效），需重新定位该字段 |

### 2.4 给 SGLang 打 V4 capture patch（**0.5.14 主路径必做**）

先看这个镜像的 sglang 是否已经自带 hook：

```bash
python -c "
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM as C
print([a for a in dir(C) if 'layers_to_capture' in a])
"
```

- 非空 → 上游已补，跳过本节
- 空 → 打 patch：

```bash
python patches/sglang/v0.5.14/apply_deepseek_v4_capture.py --check   # 只校验，6 个锚点必须全中
python patches/sglang/v0.5.14/apply_deepseek_v4_capture.py           # 落盘，自动留 .specforge-orig 备份
```

`--check` 报 `anchor matched 0 times` 说明镜像里的 `deepseek_v4.py` 与我们对过的
版本不同源。**不要用 `patch -f` 之类强上**，把没命中的锚点名记下来重对。
反悔用 `--revert`。

patch 做了 6 处改动，其中三处是 V4 特有、不能照抄 `deepseek_v2.py` 的：

1. fused mHC 下层返回的是 `[T, H]` 欠账中间量，捕获点要先补 `hc_post` 才是真实层输出
2. 载体是 `[T, hc_mult, H]` 多流，每层折成 `[T, H]` 后再拼接，才能得到 12288
3. `LogitsProcessor._get_hidden_states_to_store` 结尾 "always prefer
   `hidden_states_before_norm`"，V4 无条件传 `pre_hc_head` 会把 aux 静默覆盖成 16384

层号**不加 +1 offset**（V2 加是因为它把 list 交给 i+1 层由其 append 自身输入；
我们是在第 i 层返回后捕获）。

### 2.5 数据准备

用与最终训练完全相同的 tokenizer / template / max-length 过滤源 JSONL，
保证存在相邻监督 token（细节见 0811 文档）。冒烟阶段 8 条即可。

### 2.6 prepare hidden 冒烟

先 source §1.5 的整段环境配方，然后：

```bash
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15   # 按 npu-smi 选空闲卡
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200

torchrun --standalone --nproc_per_node=8 \
    scripts/prepare_hidden_states.py \
    --strategy dspark \
    --target-model-path /sharenfs/DeepSeek-V4-Flash-0731-w8a8 \
    --draft-model-config configs/deepseek-v4-flash-dspark.json \
    --data-path <过滤后的 jsonl> \
    --output-path ./cache/hidden_states/dsv4_smoke \
    --chat-template deepseek-v4 \
    --max-length 4096 \
    --minimum-valid-tokens 10 \
    --num-samples 8 \
    --batch-size 1 \
    --tp-size 8 \
    --sglang-attention-backend dsv4 \
    --sglang-quantization modelslim \
    --sglang-disable-radix-cache \
    --cache-dir ./cache \
    --trust-remote-code
```

要点：

- **不要加 `--compress`**，否则训练启动和读取要持续付 gzip 解压成本
- `--sglang-disable-radix-cache`：离线每 batch 新建 `RadixCache`，跨 batch 命中率为零
- `--sglang-mem-fraction-static` 不用传，见 §0 说明
- `--sglang-attention-backend dsv4`：#29794 的启动命令用的就是 `dsv4`。传
  `ascend` 也会被 `deepseek_v4_hook` 覆盖成同一个，但显式写更清楚
- `--sglang-quantization modelslim`：W8A8 checkpoint 的 `config.json` 里没有
  `quantization_config`，不显式声明就靠 SGLang 猜。#29794 的启动用的是
  `--quantization modelslim`

**起来后先确认这两行**：

```
hybrid sliding-window target: raising max_total_tokens 4096 -> 42240 ...
DSV4 pool sizes: full=42240, swa=4224, ...
```

第二行的 `swa` 必须 ≥ `batch × max_length`，否则会在 `prepare_for_extend` 挂。
对照关系：

| 请求 tokens | max_total_tokens | swa 子池 |
| --- | --- | --- |
| 2048 | 21760 | 2176 |
| 4096 | 42240 | 4224 |
| 32768 | 328960 | 32896 |

### 2.7 形状验收（采完立刻做）

```bash
python - <<'PY'
import glob, torch
f = sorted(glob.glob('./cache/hidden_states/dsv4_smoke/**/*.ckpt', recursive=True))[0]
d = torch.load(f, map_location='cpu')
for k, v in d.items():
    print(k, tuple(v.shape) if hasattr(v, 'shape') else type(v))
PY
```

硬性要求：

- `hidden_states[-1] == 12288`（3 × 4096）
- `target_last_hidden_states[-1] == 4096`（必须是全局 `hc_head` 之后；
  原始 `[B,S,4,H]` 会被 normalizer 拒绝）

出 16384 = patch 第 3 处没生效；出 4096 = 只采到一层。

**再查一遍 NaN/Inf**（W8A8 溢出不会报错，见 §1.5）：

```bash
python - <<'PY'
import glob, torch
bad = 0
for f in sorted(glob.glob('./cache/hidden_states/dsv4_smoke/**/*.ckpt', recursive=True)):
    d = torch.load(f, map_location='cpu')
    for k, v in d.items():
        if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
            print("NON-FINITE", f, k, "nan=", torch.isnan(v).sum().item(),
                  "inf=", torch.isinf(v).sum().item())
            bad += 1
print("files scanned, non-finite tensors:", bad)
PY
```

出现 non-finite 就回去确认 `INF_NAN_MODE_FORCE_DISABLE=1` 生效了。

顺带跑一次特征可用性扫描，避免训练中途才 `ValueError`：

```bash
python scripts/scan_offline_feature_eligibility.py \
    --hidden-states-path ./cache/hidden_states/dsv4_smoke --max-length 4096
```

### 2.8 数值验收（**最关键，也是唯一无法静态核验的一环**）

形状对 **不等于** 数值对。当前实现把每层的 mHC 多流做**算术平均**
（`captured.mean(dim=1)`），这是对齐 SpecForge normalizer 的假设
（`_project_target_hidden` 做 `mean(dim=-2)`），**不是从官方代码验证出来的**。
mHC 流的算术平均与 `hc_head`（带学习参数的 sigmoid 加权和）不是同一个运算。

必须做：

1. 同一条短 prompt，把 capture 的 40/41/42 层特征与官方 target forward 逐点比，
   记录每层 BF16 最大误差和 cosine
2. 确认三层在 hidden 维上的**拼接顺序**严格等于 `dspark_target_layer_ids`
   （normalizer 只校验最后一维是 12288，不校验顺序）
3. 单独验证 stream mean 折叠与官方折叠逐点一致，对着官方
   `inference/model.py` 里 `mtp.*` 的输入来源核

**这一步不过，后面权重加载、单步、收敛测试全部没有意义**——会出现
"形状对、训练能跑、loss 会降，但学的不是那个目标"，且没有任何报错提示。

如果发现官方吃的是 `hc_head` 折叠而非 mean，改动点只有一处：applier 里
`aux_hidden_states.append(captured.mean(dim=1))`。

### 2.9 训练冒烟

改 `examples/configs/deepseek-v4-flash-dspark-offline-npu.yaml` 里的
`data.hidden_states_path`、`model.draft_checkpoint_path` 和 `output_dir`。

```bash
# 先看 plan，不启进程
SPECFORGE_DEVICE=npu specforge train \
    -c examples/configs/deepseek-v4-flash-dspark-offline-npu.yaml --plan
```

期望：8 trainer ranks、EP=8、backend 为 HCCL。

#### checkpoint 是 ModelSlim W8A8，不是官方 FP4/FP8

`/sharenfs/DeepSeek-V4-Flash-0731-w8a8` 是华为 ModelSlim 量化的产物，与
SpecForge 原本适配的官方布局是两套：

| | 官方 DSpark | 这份 W8A8 |
| --- | --- | --- |
| 索引文件 | `model.safetensors.index.json` | `quant_model_weights.safetensors.index.json` |
| scale 命名 | `.weight_scale_inv` | `.weight_scale` + `.weight_offset` |
| 量化 | FP4 双 nibble / FP8 128×128 block | per-channel INT8，`scale` 形状 `[out,1]` |
| `mtp.*` 张量数 | 4705 | 7022（多出的是 scale/offset） |

**曾经的静默损坏风险**：ModelSlim 的权重 dtype 也是 int8。如果只把 scale 改名让
loader 找得到，`dequantize_v4_weight` 会走 FP4 分支、把它按"一字节两个 4-bit"
拆开——出来是垃圾，而且不报错。所以格式判别不能靠 dtype。

loader 现已支持（`aabbfa2`）。公式由三条独立证据互相印证：

1. checkpoint 自带的 `DeepSeek-V4-Flash-DSpark_best_practice.yaml` 写着
   `weight: {scope: per_channel, dtype: int8, symmetric: true}` —— 对称，
   即 `W = q * scale`，offset 恒 0
2. 抽样 offset 张量全部 `absmax=0`
3. SGLang 的 NPU W8A8 路径只把 `weight_scale` 喂给 `npu_quant_matmul`，
   全树没有一处读 `weight_offset`

格式检测沿用 SGLang 自己的规则——目录里有没有 `quant_model_description.json`
（`ModelConfig._find_quant_modelslim_config`）。该文件逐张量标注 `FLOAT` /
`W8A8_DYNAMIC`，决定哪些反量化、哪些原样加载。遇到 MXFP 系列标注会**报错而不是
猜**，offset 非零同样报错。

> 配方里 `process: - type: quarot` 的旋转是**离线折叠进权重**的：ModelSlim 包内
> 没有任何旋转代码，SGLang 对这些线性层也不做运行时变换。同一份 checkpoint 采出
> 的特征正常，就是这一点的经验证据。

**一个要记住的代价**：warm start 从 INT8 量化再反量化的草稿权重出发，不是原始
BF16 drafter，有量化损失。考虑到最终也以 W8A8 serving，这个起点是自洽的，
但它不等于官方 drafter。

#### 冻结 target embedding / head 的键名要改

DSpark 的草稿模型没有自己的 embedding 和 output head——开发文档 §2.1 写明
"冻结并复用 target 的 embedding/lm_head，只优化 `mtp.*`"。所以训练时
`TargetEmbeddingsAndHead.from_pretrained(cfg.model.target_model_path, ...)`
会去 target 目录按 `embedding_key` / `lm_head_key` 取这两个张量，**取不到就抛错**。

配置默认值是 HF 命名，而这份 checkpoint 是运行时命名：

| | 配置默认 | 这份 checkpoint |
| --- | --- | --- |
| embedding | `model.embed_tokens.weight` | `embed.weight` |
| lm head | `lm_head.weight` | `head.weight` |

两个都标注为 `FLOAT`，不需要反量化。`tie_word_embeddings: False`，两个键都要。

**这两个别名现在会自动解析**（`85a42c2` 之后）：`TargetEmbeddingsAndHead` 发现
配置的 HF 名不在索引里时，会回退到对应的运行时名并打印一行说明。所以下面这两行
是可选的，写上更明确；不写也能起来。

```yaml
model:
  target_model_path: /sharenfs/DeepSeek-V4-Flash-0731-w8a8
  draft_checkpoint_path: /sharenfs/DeepSeek-V4-Flash-0731-w8a8
  embedding_key: embed.weight
  lm_head_key: head.weight
  trust_remote_code: true
```

> 审计报告里的 `unexpected: ["mtp.0.embed.weight", "mtp.2.head.weight"]` 是**预期的**，
> 同一个原因：官方 drafter 自带这两个张量，SpecForge 的草稿模型不声明它们。
> 判定看 `missing` 和 `shape_mismatches` 是否为空。
>
> 顺带一个未排查项：官方 drafter 自带的 `mtp.0.embed` / `mtp.2.head` 与主模型的
> `embed.weight` / `head.weight` 未必是同一份权重。若不同，warm start 出来的
> drafter 会配上它训练时没用过的 embedding。要排掉就直接比这两组张量的形状和数值。

#### 权重审计（可以和 prepare 并行做，不占卡）

```bash
python scripts/audit_deepseek_v4_dspark_checkpoint.py \
    --local-dir /sharenfs/DeepSeek-V4-Flash-0731-w8a8 \
    --draft-config configs/deepseek-v4-flash-dspark.json \
    --summary-only
```

输出里先看 `checkpoint_format`：

- `modelslim` → 走 ModelSlim 校验路径，逐项应为 INT8 权重、每输出通道一个正有限
  scale、`max_abs_offset` 为 0、反量化结果有限
- `official_fp4_fp8` → 期望 4705 tensors、集合/形状零差异、FP4/FP8 sample exact

单步冒烟（4 anchors、1 objective chunk）：

```bash
SPECFORGE_DEVICE=npu specforge train \
    -c examples/configs/deepseek-v4-flash-dspark-offline-npu.yaml \
    training.max_steps=1 training.num_anchors=4 training.objective_chunk_blocks=1 \
    training.log_interval=1 training.max_checkpoints=1 \
    output_dir=/tmp/dsv4_smoke/output
```

**`training.log_interval=1` 不能省。** recipe 里是 `log_interval: 20`，而指标只在
`global_step % log_interval == 0` 时输出（`controller.py:682`）——跑不到 20 步就
**一条 loss 都不会打印，TensorBoard 里也没有**，整轮白跑。

**`output_dir` 指向本地盘。** 每个 rank 的 checkpoint 约 10.5–11 GB，8 卡一次写
约 86 GB。写 NFS 会在收尾时报：

```
RuntimeError: [enforce fail at inline_container.cc:668] unexpected pos X vs Y
```

**这不是训练失败，是短写。** 实测签名：8 个 rank 全部失败，"期望 vs 实际"差
100 或 104 字节（正好一个 zip local file header），盘上文件比失败位置多正好
32 MiB（错误抛出后页缓存继续回刷的部分）。

根因在挂载参数——`mount | grep <share>` 若出现 **`soft`**：

```
nfs vers=3, soft, timeo=100, retrans=2, local_lock=all, nolock
```

`soft` 在超时后让 I/O 失败并把短写返回给应用（`hard` 则一直重试）。
`timeo=100` + `retrans=2` 约 20–30 秒就放弃，而 86 GB 并发写必然让服务器停顿。
根治要管理员改成 `hard` 挂载；`soft` 对写入本来就不安全。

失败会留下 `.tmp`（约 86 GB 无用数据），记得 `rm -f <output>/*/*.tmp`。
`_atomic_save` 是先写 `.tmp` 再 `os.replace`，所以不会留下看似完整的坏 checkpoint。

最后一次保存是无条件的（`trainer.py:573`），没有开关能跳过。

期望：forward/backward/step 完成，loss 与所有 grad finite。

收敛冒烟：固定 32–128 条样本跑 ≥100 steps，CE / L1 / confidence 都下降，
`tau_probabilistic` 不退化。

### 2.10 A3 一期的四条盲区

单机 8 卡 = EP8/DP1 跑通，**不能**消除二期风险：

1. `draft_dp` 组大小为 1，FSDP 退化为 `NO_SHARD`，参数分片 / all-gather /
   跨 rank 梯度归约整条路径没有运行
2. checkpoint 走"每 rank 都是自己 draft-DP 组 leader"的分支，二期是另一条代码路径
3. loss 分母里的 `data_parallel_size` 一期为 1
4. 8 张卡消费同一个 batch，不是 8 路数据并行，**吞吐 ≈ 单卡，不要据此外推二期**

---

## 3. Stage 2：A2 / tige 多机

前提：Stage 1 全部通过，尤其 §2.8 的数值验收。

### 3.1 与 A3 的差异

| 项 | A3 单机 | A2 / tige |
| --- | --- | --- |
| 镜像 | `...lmsysorg/sglang:cann9.0.0-a3-v0.5.16` | `...lmsysorg/sglang:cann9.0.0-910b-v0.5.16` |
| 卡 | 910C 64 GB × 8 | 910B3，**每卡显存需实测确认** |
| prepare 拓扑 | 单机 TP=8 | 优先仍走单节点 TP=8；权重 36.99 GB/rank 是 A3 实测值，A2 上要重测 |
| 训练拓扑 | `nnodes: 1`, EP=8, DP=1 | `nnodes: N`，EP/DP 组合按显存重定 |
| 通信 | 单机 HCCL | 跨节点 HCCL，需 `MASTER_ADDR` / `NODE_RANK` |

### 3.2 prepare hidden

**先确认单节点能不能放下**：`36.99 GB × (8 / 每节点卡数)`。能放下就沿用
§2.6 的命令，一个字都不用改——采特征是纯离线的，多机没有收益，反而引入
跨节点 KV 池和 HCCL 变量。

放不下才考虑跨节点 TP，那时 `--tp-size` 要等于总卡数，
`torchrun --standalone` 换成显式 `--nnodes/--node-rank/--master-addr`。

### 3.3 训练

改 `examples/configs/deepseek-v4-flash-dspark-offline-npu.yaml`：

```yaml
deployment:
  mode: local_colocated
  trainer:
    nnodes: <N>
    nproc_per_node: <每节点卡数>
training:
  expert_parallel_size: <EP>
```

启动时每个节点传自己的 `--node-rank`（或设 `NODE_RANK` 环境变量），
`specforge train` 的 `launch_plan` 会据此展开 torchrun 参数：

```bash
SPECFORGE_DEVICE=npu specforge train \
    -c examples/configs/deepseek-v4-flash-dspark-offline-npu.yaml \
    --node-rank <i> --plan          # 先 plan
```

EP 选择：EP=8 是**最低专家显存**的配置，不是吞吐配置。显存允许时优先
EP=4/DP=2 或 EP=2/DP=4，能真正拿到数据并行。

**checkpoint 不能跨 EP 拓扑续训**：contract 里带 `expert_parallel_size`，
换拓扑前要先做 weights-only consolidation，再开新 run。

### 3.4 A2 上必须重做的验收项

不能拿 A3 的结果代替：

- 权重加载显存（§2.6 起来后看 `Load weight end` 的 `mem usage`）
- KV 池的 `DSV4 pool sizes`（ratio / page_size 可能随 backend 不同）
- §2.8 的数值验收（不同 kernel 实现，数值可能不同）
- 单 rank NPU peak memory / step time / HCCL time

---

## 4. 失败速查

| 报错 | 真实原因 | 处理 |
| --- | --- | --- |
| `no attribute 'npu_sparse_attn_sharedkv_metadata'` / `'npu_hc_pre'` | 缺 `custom_ops` | §1，换官方镜像 |
| `does not expose SGLang capture hook 'set_dspark_layers_to_capture'` | sglang 的 V4 没接 aux capture | §2.4 打 patch |
| `Prefill out of memory. Available tokens: 128` | SWA 子池被压到 10% | 已修；确认 §2.6 那两行日志出现 |
| `Not enough memory. Please try to increase --mem-fraction-static` | **误导性报错**，与真实 OOM 无关 | 别调 mem-fraction，看 §0 |
| `The memory capacity is unbalanced` | 选中的卡被别的进程占了 | `npu-smi info` 换卡 |
| normalizer `DSpark target hidden width is 16384, expected 12288` | patch 第 3 处未生效，aux 被 `pre_hc_head` 覆盖 | 重打 patch |
| 训练中途 `ValueError` 无相邻监督 token | 旧 feature 未过同一 predicate | `scripts/scan_offline_feature_eligibility.py` |
| 特征里出现 NaN/Inf，无报错 | W8A8 溢出 | `INF_NAN_MODE_FORCE_DISABLE=1`，见 §1.5 |
| `MQALayer.__init__` 断言失败 | modelslim W8A8 无 blockwise-FP8 scale | `SGLANG_OPT_FP8_WO_A_GEMM=0` |
| `RuntimeError: Could not find CUDA installation`（栈里有 `hash_topk` / `jit_kernel` / `tvm_ffi`） | MoE 路由默认走 JIT 融合 kernel，tvm_ffi 的 ninja 生成器无条件找 CUDA | `SGLANG_OPT_USE_FUSED_HASH_TOPK=False`，改走 `_forward_torch` |
| 其它 `Could not find CUDA` / `Cannot detect CUDA architecture` | 又踩到一条 GPU-only 分支 | §1.5 整段配方，不要只挑几条 |
| `HCCL function error ... hcclCommInitRootInfoConfig` + `EI0020 ... port 16666 have already been bound` | 上一次运行的进程没退干净，还占着 NPU socket | `pkill -9 -f specforge; pkill -9 -f torchrun`，`sleep 10`，`npu-smi info` 确认无残留进程；仍冲突则 `export HCCL_NPU_SOCKET_PORT_RANGE="26666-26700"` |
| `ImportError: cannot import name 'compute_dp_attention_local_info'` | 在 0.5.16 环境用了 `dsv4-dspark-moe` 分支 | 要么装算子回到 0.5.14（§1.3），要么切 `dsv4-dspark-moe-0516` |
| `ImportError: ... 'get_attention_tp_group'` | 同上 | 同上 |
| `ModelRunner.__init__() got an unexpected keyword argument 'tp_rank'` | 同上 | 同上 |
| `apply_deepseek_v4_capture.py --check` 报 anchor 不匹配 | 在 0.5.16 树上跑了 0.5.14 的 applier | 0.5.16 自带 hook，不需要打，见 §2.2 |

---

## 5. 未验证项清单

按风险排序。前两项不消除，不要投入长跑资源。

1. **mHC 折叠语义**（§2.8）。`mean(dim=1)` 未与官方核对。错了不会报错。
2. **`custom_ops` 是否真在官方镜像里**（§1.3）。推断，未验证。
3. **0.5.16 适配**（§2.2）。已对着容器真实代码树静态核对（38 个 import 全通过），
   但本地无 NPU 无 sglang，一行都没实跑过。残留风险：0.5.16 新增的
   decode-context-parallel 组，我们的 `initialize_model_parallel` 重实现没建；
   `--dcp-size` 默认 1 时应无影响。
4. **A2 平台的一切**：显存、kernel 数值、多机 HCCL、EP 拓扑。
5. **capture 的层顺序**。normalizer 只校验宽度 12288，不校验顺序。
6. **二期多机的四条盲区**（§2.10）。
7. **性能**。portable attention/MoE 是 correctness-first 实现，没有融合 NPU kernel。
9. **ModelSlim 反量化**（§2.9）。公式有规格、抽样、SGLang 实现三方印证，并有端到端
   单测（写一份小 ModelSlim checkpoint 再读回），但**没有在真机上加载过真实的
   7022 个 `mtp.*` 张量**。第一次 warm start 要盯审计输出和单步 loss 是否有限。
8. **附录 A 的算子安装流程**（当前主路径的关键一步）。从 `docker/npu.Dockerfile`
   反推，一步都没实际执行过。A.5 的拷贝法尤其存疑：vendor 包可能在
   `opp/vendors/config.ini` 里登记，单纯拷目录未必生效。


---

## 附录 A：把 NPU 算子搬进非官方容器

**先纠正一个前提：不存在"0.5.14 版本的算子"，也不需要为某个 sglang 版本编译。**
看 release 的文件名维度就清楚了：

```
custom-ops-${SGL_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN}-${DEVICE}-${arch}.zip
```

`sgl-kernel-npu` 的 tag、torch 版本、CANN 版本、device、arch —— **没有 sglang 版本**。
算子是独立于 SGLang 发布的二进制，SGLang 只是调用方。所以这件事的正确描述是
"把一份现成的算子包装进旧容器"，不是"编译一版 0.5.14 的算子"。

### A.1 这是主路径，不是权宜之计

**这是当前的主路径**（见 §0「路线选择」）：0.5.14 上除了算子之外一切都已真机验证，
换 0.5.16 反而要把验证过的 backend 换成未验证的移植。

要注意的两点：

1. **0.5.14 的 sglang 没有 `set_dspark_layers_to_capture`**，要打 §2.4 的
   applier，分支用 `dsv4-dspark-moe`。这一步这轮已经做过，不是新增工作量。
2. **算子签名漂移无法事先验证。** 算子包按 `sgl-kernel-npu` 的 tag 发布，
   与旧容器里 0.5.14 的调用点不是同一时间线。名字对得上、签名对不上时报的是
   运行期 `TypeError`，不是干净的缺失错误。0.5.16 树一共调用 9 个算子：
   `npu_hc_pre` / `npu_hc_post` / `npu_sparse_attn_sharedkv` /
   `npu_sparse_attn_sharedkv_metadata` / `npu_quant_lightning_indexer` /
   `npu_quant_lightning_indexer_metadata` / `npu_mla_prolog_v3` /
   `npu_moe_gating_top_k` / `inplace_partial_rotary_mul` / `compressor`。
   旧树调用的可能是其中一个子集，或同名不同签名。

### A.2 先读旧容器自己的 Dockerfile

0.5.14 容器的 sglang 也是 NPU DSV4 fork，大概率带着构建它的 Dockerfile，
里面写死了它期望的 kernel 版本：

```bash
grep -nE "SGLANG_KERNEL_NPU_TAG|CANN_VERSION|DEVICE_TYPE|sgl-kernel-npu" \
    /sgl-workspace/sglang/docker/npu.Dockerfile
```

0.5.14 容器实测三个值都是默认：`CANN_VERSION=9.0.0`、`DEVICE_TYPE=a3`、
`SGLANG_KERNEL_NPU_TAG=main`。**但 `main` 不是真实 tag**，见 A.4。

### A.3 四要素必须对得上

算子包是按这四个维度构建的，任何一个不匹配都不要装：

```bash
# 在旧容器里执行
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null | head -3   # CANN 版本
python -c "import torch, torch_npu, sys; print(torch.__version__, torch_npu.__version__, sys.version)"
arch                                                                            # aarch64 / x86_64
npu-smi info | head -5                                                          # 确认 a3 还是 910b
```

需要满足：CANN **9.0.0**、torch/torch_npu **2.10.0**、Python **3.11**（cp311）、
arch 一致、DEVICE_TYPE 一致。旧容器已知是 torch_npu 2.10.0 + py3.11.15，
**CANN 版本和 arch 是未知项，必须先查**。

### A.4 选 tag（有实测数据）

`docker/npu.Dockerfile` 里的 `SGLANG_KERNEL_NPU_TAG=main` **不是真实 tag**。
`sgl-project/sgl-kernel-npu` 的 release 是日期版本号，且有两条硬事实：

- **`custom-ops-*` 和 `ops-transformer-*` 从 `2026.7.0`（2026-07-02）才开始发布。**
  `2026.6.2` 及更早的 release 只有 `sgl-kernel-npu-*` 一个包。
- 0.5.14 容器里装的是 `sgl_kernel_npu 2026.5.1`，即 tag **`2026.05.01`**
  （pip 按 PEP 440 规范化后是 `2026.5.1`）。**那一版还没有 custom-ops。**
  所以容器缺 `custom_ops` 不是漏装，是构建时它尚不存在。

可用范围是 `2026.7.0` 起。截至 2026-08-18 带 `cann9.0.0-a3` 和 `cann9.0.0-910b`
构建的 tag：

```
2026.7.0  2026.7.1  2026.7.2  2026.7.4  2026.7.27  2026.8.10  2026.8.13  2026.8.17
```

`cann9.1.0` 的构建是 py312，**不要用**——容器是 py3.11。`cann9.0.0` 线才是 py311。

**选哪个**：挑与容器里 sglang 源码树时间最接近的，算子签名漂移的概率最小。

```bash
git -C /sgl-workspace/sglang log -1 --date=short --format='%ad %h %s'
```

镜像是以保留 git 历史的方式 clone 的，这条能给出源码树的日期。拿这个日期去对上面的
tag 列表，取**不晚于它**的最近一个；没有更早的就从 `2026.7.0` 起试。
出现 `TypeError` 类的签名不匹配时，沿列表向前后各挪一档重试。

### A.5 做法一：下载并安装

以 A3 / aarch64 / CANN 9.0.0 / tag `2026.7.27` 为例——**三个变量按你自己的实测值替换**：

```bash
# ---- 0. 先核实四要素，任何一项对不上就不要继续 ----
arch                                                    # aarch64 还是 x86_64
python -c "import torch, torch_npu, sys; print(torch.__version__, torch_npu.__version__, sys.version.split()[0])"
ls -d /usr/local/Ascend/cann-*/opp 2>/dev/null || ls -d /usr/local/Ascend/ascend-toolkit/*/opp
```

要求：torch/torch_npu **2.10.0**、Python **3.11**、CANN 目录里有 **9.0.0**。
最后一条同时给出 `.run` 的 `--install-path`——**用实际存在的那个路径**，
不要照抄 `cann-9.0.0`。

```bash
# ---- 1. 下载 ----
TAG=2026.7.27
CANN=9.0.0
DEVICE=a3                      # A2/910B 机器改成 910b
A=$(arch)
OPP=/usr/local/Ascend/cann-${CANN}/opp        # 换成上一步实测到的路径
BASE=https://github.com/sgl-project/sgl-kernel-npu/releases/download/${TAG}

mkdir -p /tmp/npu-kernels && cd /tmp/npu-kernels
wget ${BASE}/custom-ops-${TAG}-torch2.10.0-cann${CANN}-${DEVICE}-${A}.zip
wget ${BASE}/ops-transformer-${TAG}-torch2.10.0-cann${CANN}-${DEVICE}-${A}.zip

# ---- 2. 解包 ----
unzip -o "custom-ops-${TAG}-torch2.10.0-cann${CANN}-${DEVICE}-${A}.zip"
unzip -o "ops-transformer-${TAG}-torch2.10.0-cann${CANN}-${DEVICE}-${A}.zip"
chmod +x *.run
ls -la                          # 应看到两个 .run 和一个 custom_ops-*.whl

# ---- 3. 装 CANN 侧算子（两个 .run）----
./CANN-custom_ops-none-linux.${A}.run        --install-path=${OPP}
./cann-ops-transformer-custom_linux-${A}.run --install-path=${OPP}

# ---- 4. 装 Python 绑定（whl）----
pip install --no-deps custom_ops-1.0-cp311-cp311-linux_${A}.whl
```

**`.run` 和 `.whl` 缺一不可**：`.run` 装的是 CANN 侧算子实现（落到
`opp/vendors/customize` 和 `opp/vendors/custom_transformer`），`.whl` 装的是
`torch.ops.custom` 的 Python 绑定。`sgl-project/sglang#29794` 的报告者只有前者，
所以仍然缺 `npu_hc_pre`。

**不要动 `sgl_kernel_npu`**：容器里 2026.5.1 那份是配套 sglang 树在用的，
第三个 zip（`sgl-kernel-npu-*`，含 DeepEP）先不要装，避免一次引入两个变量。
只有在 `custom_ops` 装好后仍报 `sgl_kernel_npu` 相关缺失时再考虑同步升级。

### A.6 离线传输（容器连不上 GitHub 时）

两个包一共约 **60 MB**（`custom-ops` 53.9 MB + `ops-transformer` 6.1 MB），
本地下好再传是完全可行的，在受限网络环境里这通常是常规做法而非退路。

浏览器或任意有 GitHub 访问的机器上下载：

```
https://github.com/sgl-project/sgl-kernel-npu/releases/tag/<TAG>
```

按 A.3 的四要素挑对应的两个文件，例如 A3 / aarch64 / CANN 9.0.0 / `2026.7.27`：

```
custom-ops-2026.7.27-torch2.10.0-cann9.0.0-a3-aarch64.zip        56,528,810 B
ops-transformer-2026.7.27-torch2.10.0-cann9.0.0-a3-aarch64.zip    6,458,012 B
```

**传输后必须校验**，二进制包截断了不会有明显症状，会以后续莫名其妙的错误出现。
每个 release 资产的 sha256 可以从 API 取：

```bash
# 有 GitHub 访问的机器上
gh api repos/sgl-project/sgl-kernel-npu/releases/tags/<TAG> \
    --jq '.assets[] | select(.name|test("cann9\\.0\\.0-a3-aarch64")) | "\(.digest)  \(.name)"'
```

例（`2026.7.27` / a3 / aarch64）：

```
sha256:f504928939fae9e65188ccda0c6cbd81c8aac3ed7383e3e505c5dd3a512f153f  custom-ops-...-a3-aarch64.zip
sha256:f6373007f19b45c8cb32e978bc21b931cef39acce1cafbd030ac66b8dfe81596  ops-transformer-...-a3-aarch64.zip
```

传进去并核对：

```bash
# 宿主机
scp custom-ops-*.zip ops-transformer-*.zip <user>@<host>:/tmp/
# 宿主机 -> 容器
docker cp /tmp/custom-ops-<TAG>-torch2.10.0-cann9.0.0-a3-aarch64.zip      <容器>:/tmp/npu-kernels/
docker cp /tmp/ops-transformer-<TAG>-torch2.10.0-cann9.0.0-a3-aarch64.zip <容器>:/tmp/npu-kernels/

# 容器内：核对后再解包
cd /tmp/npu-kernels && sha256sum *.zip
```

两个 sha256 与上面一致后，回到 A.5 的第 2 步继续（解包、两个 `.run`、一个 `.whl`）。

### A.7 做法二：从官方镜像里拷

A.5/A.6 的 release 路线走不通时的退路。前提是你已经拉到了官方
v0.5.16 镜像——那套组合必然自洽，但**它是按 CANN 9.0.0 构建的**，旧容器 CANN 版本
不同就不能这么拷。

镜像的安装脚本装完就把 zip 删了（`rm -rf cann-custom-ops`），所以要拷的是
**已安装的产物**，三处：

```bash
# 在官方 v0.5.16 容器里，先确认路径
python -c "import custom_ops, os; print(os.path.dirname(custom_ops.__file__))"
ls -d /usr/local/Ascend/cann-9.0.0/opp/vendors/customize
ls -d /usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer
```

三份都打包带走（Python 绑定 + 两个 CANN vendor 包）：

```bash
# 官方容器内
tar czf /tmp/npu-kernels.tgz     -C "$(python -c 'import custom_ops,os;print(os.path.dirname(os.path.dirname(custom_ops.__file__)))')" custom_ops     -C /usr/local/Ascend/cann-9.0.0/opp/vendors customize custom_transformer

# 宿主机
docker cp <官方容器>:/tmp/npu-kernels.tgz .
docker cp npu-kernels.tgz <旧容器>:/tmp/

# 旧容器内：解到对应位置
tar xzf /tmp/npu-kernels.tgz -C /tmp/unpack --one-top-level
cp -r /tmp/unpack/custom_ops "$(python -c 'import site;print(site.getsitepackages()[0])')/"
cp -r /tmp/unpack/customize /tmp/unpack/custom_transformer /usr/local/Ascend/cann-9.0.0/opp/vendors/
```

`custom_ops` 是编译好的扩展，**必须 cp311 + 同 arch**，否则 import 就失败。

### A.8 装完的验证

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

python - <<'PY'
import torch, custom_ops  # noqa: F401
need = [
    "npu_hc_pre", "npu_hc_post", "npu_sparse_attn_sharedkv",
    "npu_sparse_attn_sharedkv_metadata", "npu_quant_lightning_indexer",
    "npu_quant_lightning_indexer_metadata", "npu_mla_prolog_v3",
    "npu_moe_gating_top_k", "inplace_partial_rotary_mul", "compressor",
]
have = set(dir(torch.ops.custom))
for n in need:
    print(("[ok]   " if n in have else "[MISS] ") + n)
PY
```

再核对旧容器的 sglang 实际调用了哪些（可能是上面的子集）：

```bash
grep -rhoE "torch\.ops\.custom\.[a-zA-Z0-9_]+" \
    "$(python -c 'import sglang,os;print(os.path.dirname(sglang.__file__))')" | sort -u
```

两边取交集为空缺项才算通过。**注意这只验证了名字，验证不了签名**——
签名不匹配要到 forward 才以 `TypeError` 暴露。

### A.9 本附录未经验证

以上步骤是从容器树里的 `docker/npu.Dockerfile` 反推的，**我没有在任何机器上执行过**。
`.run` 安装器的实际参数、拷贝法能否绕过 `.run` 的注册逻辑（vendor 包可能在
`opp/vendors/config.ini` 里登记，单纯拷目录未必生效）都需要实测。
拷贝法失败时退到做法二。
