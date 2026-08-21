# DeepSeek-V4 DSpark MoE 昇腾训练适配开发文档

- 日期：2026-08-12
- 基线：`0812_upstream@652350a`
- 开发分支：`dsv4-dspark-moe`
- 当前状态：代码实现与无设备测试完成，等待 Ascend NPU 穿刺

## 1. 结论

本分支补上的不是“让原有 Qwen DSpark 配置接受 DeepSeek-V4 的名字”，而是一个
独立的、与官方权重同构的 `DeepseekV4DSparkDraftModel`。它能够：

1. 从官方 `DeepSeek-V4-Flash-DSpark` 融合 checkpoint 中只加载
   `model.mtp.0/1/2` 草稿权重；
2. 识别 HF 原始命名与未分片（MP=1）官方推理 runtime 命名，并把 FP4
   专家、FP8 投影的
   E8M0 block scale 解量化成可训练的 BF16 参数；
3. 在 portable PyTorch eager/SDPA 路径上执行三阶段 mHC + V4 attention + MoE
   的前向和反向，不依赖 CUDA、Triton、FlashInfer 或 TileLang；
4. 用 `expert_parallel_size` 在 HCCL rank 间切分 256 个 routed experts，保持
   router、shared expert、attention 和 mHC 参数复制；
5. 复用 SpecForge 现有 DSpark 的 CE + 分布 L1 + confidence BCE 目标，以目标
   V4 的选定层 hidden state 和末层 hidden state 微调公开 drafter；
6. 保存每个 EP rank 的本地专家、原地 resume，并在 warm start / export 前合并
   完整草稿 state dict。

这已经形成“官方权重 -> 离线特征 -> Ascend EP 微调 -> checkpoint 合并”的代码
闭环。但是开发机没有 Ascend NPU，本轮没有声称完成真机 HCCL、显存、吞吐或收敛
验证；第 9 节给出了必须执行的上机验收。

## 2. 先把模型认清：它不是普通的三层小 Transformer

### 2.1 官方来源交叉核对

架构以以下一手资料为准：

- [官方 Flash DSpark 推理配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/inference/config.json)
  明确给出 `n_mtp_layers=3`、`dspark_block_size=5`、目标层 `[40,41,42]`、
  256 routed experts / top-6、一个 shared expert、SWA window 128；
- [官方推理模型代码](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/inference/model.py)
  是本实现的结构与张量方向基准；
- [官方 checkpoint 转换器](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/inference/convert.py)
  给出了 HF/runtime 的精确重命名、FP4 nibble 顺序和 expert shard 规则；
- [DSpark 论文](https://arxiv.org/abs/2607.05147) 给出半自回归 block、Markov
  head 与 confidence-scheduled verification 的训练动机。

官方模型卡说明 DSpark 文件是同一个 V4 checkpoint 外加 speculative module，
不是一份独立目标模型。因此训练时冻结并复用 target 的 embedding/lm_head，只优化
`mtp.*`，与 checkpoint 组织方式一致。

### 2.2 三阶段真实结构

Flash drafter 的关键尺寸如下：

| 项目 | 值 |
| --- | --- |
| hidden / target depth | 4096 / 43 |
| DSpark stage | 3（`mtp.0`、`mtp.1`、`mtp.2`） |
| target capture | 40、41、42 层，拼成 12288 维后投回 4096 |
| 并行草稿长度 | 5 |
| mHC streams | 4 |
| attention | shared latent KV、64 heads、head dim 512、partial RoPE 64、SWA 128 |
| Q/O low rank | Q rank 1024；8 个 O groups、O rank 1024 |
| MoE | 每 stage 256 routed + 1 shared，top-6，expert hidden 2048 |
| router | `sqrt(softplus(score))` + correction bias top-k，route scale 1.5 |
| Markov / confidence | rank 256；confidence 输入为 pre-norm head hidden + Markov embedding |

每个 stage 都是完整 V4 block，而不是“一个小 attention 再接一个 head”：

```text
selected target hidden (40/41/42, mHC stream mean)
  -> mtp.0.main_proj + main_norm

anchor token + 4 noise tokens
  -> expand to 4 mHC streams
  -> [mHC -> V4 shared-latent SWA -> mHC -> routed MoE + shared expert] x 3
  -> mtp.2.hc_head
  -> pre-norm hidden ---------------------> confidence head (+ Markov embedding)
  -> mtp.2.norm -> frozen target lm_head -> Markov bias -> 5 token distributions
```

这里有四个容易误实现的细节：

1. 目标 V4 层输出是 `[B,S,4,H]` mHC streams，官方送入 DSpark 的是
   `mean(dim=2)`；三个目标层再沿 hidden 维拼接。
2. mHC `comb` 的 post contraction 是
   `sum(comb[..., i, j] * residual[..., i, :], i)`，等价于转置后矩阵乘，不能写成
   普通的 `comb @ residual`。
3. DSpark query 可以同时看 anchor、最近 128 个 target states 和本 block 全部 5
   个并行 draft token；它不是块内因果 attention。
4. confidence head 使用 `hc_head` 的 pre-norm hidden，而 vocab logits 先过 final
   RMSNorm；两条 head 输入不能共用一个“顺手 norm 后”的张量。

## 3. 我们怎样微调它

### 3.1 冻结 target，只训练 drafter

目标模型在训练中承担三个角色：

- 从第 40/41/42 层提供 `main_hidden`；
- 提供末层预测 hidden，经过冻结 lm_head 得到 teacher distribution；
- 提供冻结 token embedding 和 lm_head，保持草稿与目标的词表语义一致。

训练参数是三层 `mtp.*` 内的 mHC、attention、router、routed/shared experts、
main projection、Markov head 和 confidence head。当前实现是全参数 drafter 微调，
不是 LoRA；它不会修改 284B target backbone。

### 3.2 目标函数

SpecForge 现有 DSpark 目标直接适用：

```text
L = alpha_ce * CE(draft_logits, target_token)
  + alpha_l1 * L1(softmax(draft_logits), teacher_distribution)
  + alpha_conf * BCE(confidence, exact_accept_probability)
```

示例保留 `0.1 CE + 0.9 L1 + 1.0 confidence`。Markov `W1` 用前一个真实 token
索引，`W2` 生成当前位置 vocab bias。confidence 监督是 draft/teacher 分布的重叠
概率，用来训练推理时的验证长度调度。

每条序列采样多个 anchor。模型前向把 anchor 折到 batch 维，只构造
`[B*num_anchor, 5, 128+5]` attention，而不是形成不同 anchor 两两相见的巨型
`[num_anchor*5]^2` 分数矩阵。这一点是 128 anchors 配置能落地的必要条件。

### 3.3 官方公开权重怎样成为可训练参数

官方 Flash mixed checkpoint 中，MoE expert 是 FP4 + 每 32 个 K 元素一个 E8M0
scale，多数投影是 FP8 + 128x128 block scale。训练路径做以下处理：

- 用官方 E2M1 table 解两个 nibble（low nibble 在前）；
- 解 E8M0 指数 scale；
- FP4 按 K 维每 32 元素展开，FP8 按 128x128 block 展开；
- 复制到本 rank 的 BF16 trainable parameter；
- FSDP1 要求同一 FlatParam 的存储 dtype 一致，因此可训参数统一以
  BF16 存储；RMSNorm、mHC、attention sink 和 confidence 计算仍在
  forward 中显式转 FP32。`gate.bias` 是 FP32 buffer，不进 optimizer。

这是“量化 checkpoint 初始化、BF16 微调”，不是 FP4 QAT。代价是训练显存高于
官方推理权重；示例默认打开 `optimizer_cpu_offload`，把 FP32 master 和 Adam
moments 放在 CPU。

## 4. 代码实现

### 4.1 新模型与配置

- `specforge/modeling/draft/deepseek_v4_dspark.py`
  - config、RMSNorm、mHC/Sinkhorn；
  - V4 shared-latent attention、partial interleaved RoPE、learned sink、grouped O-LoRA；
  - `sqrtsoftplus` router、SwiGLU experts、shared expert；
  - 三阶段 drafter、Markov/confidence heads；
  - FP4/FP8 dequant 与官方 fused checkpoint loader。
- `scripts/audit_deepseek_v4_dspark_checkpoint.py`
  - 只读 index 和含 `mtp.*` 的 3 个 shard header；
  - 对本实现 state 集合/形状做 diff；
  - 用 HTTP Range 抽取 FP4/FP8 真实 tensor，不下载整个 shard。

远程 warm start 也采用同样的最小下载契约：每个节点只由 local-rank 0
下载 3 个 `mtp.*` shard，以及 embedding/lm_head 实际所在的 shard；
其他 rank 在分布式同步后只读本节点 HF cache。
- `configs/deepseek-v4-flash-dspark.json`
  - 与官方 Flash DSpark 推理配置对齐的本地 draft config。

官方 `config.json` 的 architecture 仍是 `DeepseekV4ForCausalLM`。只要同时存在
`dspark_block_size` 和 `dspark_target_layer_ids`，loader 会把它转换成注册的
`DeepseekV4DSparkDraftModel`，而不会错误实例化整套 target。

### 4.2 数据与 attention 契约

- 离线 normalizer 新增 target selected-layer 的
  `[1,S,L,4,H]` / `[1,S,4,H]` 原生 V4 mHC capture；
- teacher `target_last_hidden_states` 必须是 target 全局 `hc_head` 之后的
  rank-2/rank-3 tensor，遇到原始 mHC streams 会早失败，不做错误的简单平均；
- DSpark mask 新增 anchor 可见和全 block 并行可见模式；
- position offset 改为官方的 `anchor + 1 ... anchor + 5`；
- teacher head 与 draft objective head 分离，避免把 drafter final norm 施加到目标
  last hidden 上。

离线 feature 契约仍是：

```text
input_ids
loss_mask
hidden_states                # target 40/41/42，期望最后为 3*4096
target_last_hidden_states    # target 最终 head 前的预测 hidden，宽 4096
```

### 4.3 Ascend expert parallel

新增 `training.expert_parallel_size`。训练 mesh 从 `(draft_dp, sp)` 变成
`(draft_dp, ep, sp)`；EP peers 消费相同样本、拥有互斥 routed experts，只有
draft-DP replicas 分不同样本并参与 loss denominator/FSDP reduction。

当前 EP 使用 replicated-token 方案：

1. 每个 rank 都算 router；
2. 只执行本 rank 的 experts；
3. HCCL all-reduce routed output；
4. shared expert 每 rank 各算一次；
5. backward 对 routed-input/router-input 梯度做 EP all-reduce；router 参数梯度也
   跨 EP 求和，本地 expert 参数梯度保持本地。

专门处理了“这个 microbatch 在某个 EP rank 上没有选中任何本地 expert”的情况：
该 rank 仍保留零值 autograd 路径并参加相同顺序的 collective，避免 HCCL 挂死。

这是 correctness-first 路径，不做 all-to-all token dispatch。对 5-token 小 block
与较多 anchor，复制 token 通常比复杂 dispatch 更容易稳定；性能仍需真机测量。

### 4.4 checkpoint

每个 EP/FSDP group leader 保存自己的本地 expert `draft_state_dict` 到
`training_state_rank{rank}.pt`。resume 根据当前 `(ep,sp)` coordinate 加载对应
leader；warm start 和 exporter 会 union 全部 rank 文件，恢复完整 `mtp.*`。

共享参数在多个 shard 文件中重复，形状/dtype 冲突会拒绝合并。checkpoint contract
新增 `expert_parallel_size`，因此不要用不同 EP 拓扑直接续训；需要换拓扑时先做
weights-only consolidation，再开启新 run。

## 5. DeepSeek-V4 模板

新增 `deepseek-v4` chat template，边界来自官方
[V4 encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/encoding/encoding_dsv4.py)：
`<｜User｜>`、`<｜Assistant｜>`、`<｜end▁of▁sentence｜>`。模板不额外注入系统
prompt，避免覆盖 V4 tokenizer 自己的编码逻辑；`<think>`/`</think>` delimiter 不
进入监督 mask。

V4 的 tool/reasoning 协议比这个 loss-boundary registry 更复杂。建议第一轮使用
普通 user/assistant SFT 对话，并以 checkpoint tokenizer 的
`apply_chat_template` 输出为准；tool 数据要先做小样本 mask 可视化再全量采集。

## 6. 使用方式

### 6.1 环境和数据过滤

先用与最终训练相同的 tokenizer、template 和 max length 过滤源 JSONL，保证存在
相邻监督 token；命令细节见 0811 文档。然后准备离线 feature：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 scripts/prepare_hidden_states.py \
  --strategy dspark \
  --target-model-path deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --draft-model-config configs/deepseek-v4-flash-dspark.json \
  --data-path /data/train.filtered.jsonl \
  --output-path ./cache/hidden_states/deepseek-v4-flash-dspark \
  --chat-template deepseek-v4 \
  --max-length 4096 \
  --minimum-valid-tokens 10 \
  --tp-size 8 \
  --batch-size 8 \
  --sglang-attention-backend ascend \
  --trust-remote-code
```

不要加 `--compress`，否则训练启动和读取要持续付 gzip 解压成本。采完必须抽一条
检查形状：`hidden_states[-1] == 12288`，`target_last_hidden_states[-1] == 4096`。
如果 selected-layer hook 给的是原生 mHC rank-4/rank-5 tensor，normalizer 会先
stream mean，仍应在 NPU 上与官方 `h.mean(dim=2)` 做数值抽检。
`target_last_hidden_states` 则必须在 target 的全局 `hc_head` 后采集；原始
`[B,S,4,H]` 形状会被拒绝，因为它既不能用 stream mean 代替，也不能直接
送入冻结 LM head。

### 6.2 先看 launch plan，再训练

本节只描述**一期：单机 8 卡 = EP8/DP1**。示例 recipe 保持
`nnodes: 1`、`nproc_per_node: 8`、`expert_parallel_size: 8`，这是当前最低专家
显存的穿刺配置，不是二期 8 节点配置。

修改 `examples/configs/deepseek-v4-flash-dspark-offline-npu.yaml` 中的数据路径、
checkpoint 本地路径和输出目录：

```bash
SPECFORGE_DEVICE=npu specforge train \
  -c examples/configs/deepseek-v4-flash-dspark-offline-npu.yaml --plan

SPECFORGE_DEVICE=npu specforge train \
  -c examples/configs/deepseek-v4-flash-dspark-offline-npu.yaml
```

统一 launcher 会按 `deployment.trainer.nproc_per_node: 8` 启动 torchrun，并在
`torch_npu` 活跃时选择 HCCL。示例是 EP=8、DP=1，每 rank 负责 32 个 routed
experts/stage。这个选项的收益是专家权重切分后更易放入 64GB NPU，
不是吞吐扩展：8 张卡消费同一 batch，attention/mHC/shared expert/router
都重复计算，吞吐约等于单卡。冒烟数据集应按这个规模选，不要用一期 step time
外推二期。

一期有四条明确盲区，单机跑通不能消除对应的二期风险：

1. `draft_dp` 组大小为 1，FSDP 实际退化为不分片，并打印
   `FSDP is switching to use NO_SHARD since the world size is 1`。参数分片、参数
   all-gather 和跨 rank 梯度归约整条路径一期完全没有运行。
2. checkpoint 写入走“每个 rank 都是自己 draft-DP 组 leader”的分支。二期
   `draft_dp=8` 时只有全局 rank 0～7 会产出 `draft_state_dict`，是另一条代码路径。
3. loss 分母中的 `data_parallel_size` 一期为 1，二期为 8。
4. 所有 8 张卡消费同一个 batch，因此一期不是 8 路数据并行。

在一期的 world=8 下，可对照的拓扑为：

| 拓扑 | 专家权重/卡 | 理论 draft-DP | 用途 |
| --- | ---: | ---: | --- |
| EP=8 / DP=1 | 32 experts/stage | 1 | 最低专家显存，吞吐可能接近或低于单卡 |
| EP=4 / DP=2 | 64 experts/stage | 2 | 显存允许时的首选平衡点 |
| EP=2 / DP=4 | 128 experts/stage | 4 | 专家显存最高，有更多数据并行机会 |

当 EP=4/2 不能放入显存时才选 EP=8；不要在同一 run 中途改 EP。
portable attention 内部是 eager `einsum`，recipe 的 `attention_backend: sdpa`
只选择兼容 mask 契约，性能和显存不应按 fused SDPA/FA 预估。

首轮穿刺建议覆盖配置：

```bash
SPECFORGE_DEVICE=npu specforge train \
  -c examples/configs/deepseek-v4-flash-dspark-offline-npu.yaml \
  training.max_steps=2 training.num_epochs=1 \
  training.num_anchors=4 training.objective_chunk_blocks=1 \
  training.accumulation_steps=1 training.save_interval=1
```

### 6.3 resume 与权重导出

原拓扑 resume：

```bash
SPECFORGE_DEVICE=npu specforge train \
  -c examples/configs/deepseek-v4-flash-dspark-offline-npu.yaml \
  training.resume_from=./outputs/deepseek-v4-flash-dspark-offline-npu/deepseek-v4-flash-dspark-offline-npu-latest
```

现有 HF exporter 能读取 EP checkpoint，先合并 rank-local experts 再物化 draft。
但完整 BF16 drafter 很大，export 需要大内存 CPU 节点。更重要的是，当前 exporter
输出的是独立 draft checkpoint；把微调后的 `mtp.*` 覆盖回原 V4 fused shard、再
量化回推理 FP4/FP8并接 vLLM-Ascend/SGLang，是下一阶段 serving adapter 工作，
本分支不把它冒充成已完成。

### 6.4 二期扩展：8 节点 64 卡

二期不复用一期 recipe 原样启动。把 `deployment.trainer.nnodes` 改成 8，并且必须
配置 `deployment.trainer.master_addr`；schema 对 `nnodes>1` 强制要求
`master_addr`，缺失时 `Config.model_validate` 会直接报错。每节点仍为
`nproc_per_node: 8`，总 world size 为 64。

world=64 时，EP=8 才会得到真正的 8 路 draft 数据并行。§6.2 的表按 world=8
计算，只适用于一期；二期应按下表重新预算：

| 二期拓扑 | 专家并行 EP | draft-DP | 说明 |
| --- | ---: | ---: | --- |
| EP=8 | 8 | 8 | 每卡 32 experts/stage，8 路数据并行 |
| EP=4 | 4 | 16 | 每卡 64 experts/stage，16 路数据并行 |
| EP=2 | 2 | 32 | 每卡 128 experts/stage，32 路数据并行 |

rank 摆放是这个配置成立的关键。mesh 维度为 `(draft_dp, ep, sp)`，且
`rank = node * 8 + local_rank`；在 EP=8、SP=1 时，每个 EP group 正好是同一节点
内连续的 8 个 rank，所以每层 MoE 的 all-reduce 走机内网络。每个 draft-DP group
则从 8 个节点各取同一 `local_rank` 的一个 rank，所以 FSDP 归约走机间网络。
不要在不了解这个约束时改变 rank 排布或 mesh 维度顺序。

二期启动前，`output_dir`、`data.hidden_states_path`、target 模型目录和官方 DSpark
checkpoint 目录必须在 8 个节点上都可见。`training_state_rank{0..63}.pt` 必须全部
写入同一个共享 checkpoint 目录；合并与 resume 都会 glob 单个目录，文件分散在
各节点本地盘会直接失败。

二期必须单独验证 `SHARD_GRAD_OP`：每个 block 的参数 all-gather 会经过机间网络，
需要对比 `SHARD_GRAD_OP` 与 `NO_SHARD` 的 step time。当前 hidden-state 采集命令是
单节点 `--nproc_per_node=8 --tp-size 8`，扩到二期前还要重新确认 target 权重形态和
TP rank 摆放，不能仅凭训练侧 world size 校验通过就假设采集拓扑也成立。

## 7. 社区参考：什么能借，什么不能当训练证据

- [NVIDIA NeMo AutoModel 的 DSpark 训练指南](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dspark-speculative-decoding)
  是最有价值的 GPU 训练参照：目标 hidden/offline data、CE/分布匹配/confidence
  多目标与 draft-only optimization 都和本路线一致；但其文档里的 V4 draft 组件
  不能替代对官方 released `mtp.*` MoE checkpoint 的结构核对。
- [DeepSpec](https://github.com/deepseek-ai/DeepSpec) 提供 Qwen/Gemma 的 DSpark
  训练框架和目标，但公开 config 并不等于 V4 三阶段 MoE 的训练实现。本分支复用
  其算法思想，不复用 GPU kernel。
- 社区所谓“fine-tune 成功”的公开例子中，当前可核实的
  [MiaAI-Lab 状态](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-Abliterated-Uncensored/blob/main/docs/STATUS_FINETUNE.md)
  实际是对 `attn.wo_b`/`mtp.wo_b` 做 refusal-direction weight projection，不是用
  DSpark teacher objective 训练整个 drafter。它证明 fused `mtp.*` overlay 与 GPU
  服务链路可操作，但不能证明完整梯度微调已经跑通。
- 大量 DGX Spark/A40/Blackwell 报告是 serving、量化或 weight surgery，适合参考
  checkpoint overlay 和运行时参数，不应当写成训练复现证据。

另一方面，[vLLM-Ascend 的 V4 教程](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/DeepSeek-V4-Flash.md)
已给出 Ascend 上 V4 expert-parallel serving 的公开路径；
[SLAI T-Rex](https://arxiv.org/abs/2607.20145) 也证明 V4 family 全参 post-training
能够在 Ascend SuperPOD 上完成。这些支持“底层算子与集群方向可行”，但仍不能替
本项目的 HCCL drafter 训练验收。

## 8. 本地验证结果

本轮有意不跑 GPU test，也没有 NPU 硬件。已完成的 device-independent 证据：

| 检查 | 结果 | 覆盖内容 |
| --- | --- | --- |
| tiny model forward/backward | 通过 | 三 stage、mHC、attention、MoE、Markov/confidence |
| tiny 完整 teacher objective | 通过 | 真实 V4 draft + 冻结 embedding/head + CE/L1/confidence 反传 |
| 官方 state key loader | 通过 | `model.mtp`、`self_attn/mlp`、gate bias 重命名 |
| 官方 metadata 审计 | 未执行 | 仓库尚无针对本地真实 checkpoint 的审计产物 |
| 真实量化 tensor 审计 | 未执行 | 仓库尚无 FP4 expert / FP8 `wo_a` 的真实 tensor 审计产物 |
| FP4 + E8M0 | 通过 | nibble 顺序与 scale 展开 |
| V4 mask | 通过 | anchor、128-window、5-token 并行可见 |
| mHC post reference | 通过 | 官方 first-axis contraction，防 transpose bug |
| 2-rank Gloo EP 等价 | 通过 | output、input grad、router grad、本地 expert grad |
| 空本地 route | 通过 | collective 顺序仍一致 |
| EP checkpoint union | 通过 | rank-local experts 合并、冲突拒绝 |
| FSDP/DDP backend 双 step | 通过 | 参数 dtype 统一；unused expert 不再破坏 DDP |
| noaux_tc | 通过 | correction bias buffer、路由计数与 optimizer-step 后非梯度更新 |
| native mHC feature normalizer | 通过 | selected-layer stream mean/layer flatten；错误 teacher streams 早失败 |
| schema / topology | 通过 | EP 只允许 DSpark，world size 可整除，online 拒绝 |
| 原 DFlash loss/sliding/normalizer CPU 回归 | 通过 | 默认行为不变；CUDA-only mask 用例未运行 |

真实 checkpoint 就绪后，审计必须使用本地模式并把完整 JSON 一并提交：

```bash
python scripts/audit_deepseek_v4_dspark_checkpoint.py \
  --local-dir /shared/models/DeepSeek-V4-Flash-DSpark \
  --draft-config configs/deepseek-v4-flash-dspark.json \
  --output my_docs/deepseek-v4-flash-dspark-checkpoint-audit.json
```

本地目录有 `model.safetensors.index.json` 时按 index 定位 DSpark shards；没有时扫描
目录下全部 `*.safetensors`。报告会列出 checkpoint 绝对路径、每个 shard 的文件名与
字节数、完整 missing/unexpected/shape-mismatch 列表，以及一个 FP4 expert 和一个
FP8 `wo_a` 的 tensor/scale 名称与逐元素最大误差。本机当前只找到带 index、无任何
权重 shard 的 6.3MB Hugging Face cache，因此上述两项保持“未执行”，没有生成或提交
虚假的通过产物。

## 9. NPU 上机验收清单

按顺序做，前一项不通过不要直接长跑：

1. **feature 数值（最先执行）**：用同一条短 prompt，把 capture 的 40/41/42 层
   特征与官方 target forward 逐点对比，记录每层 BF16 最大误差和 cosine。必须确认
   三层在 hidden 维上的拼接顺序严格等于 `dspark_target_layer_ids`，并单独验证
   rank-4/rank-5 输入的 stream mean 折叠与官方折叠逐点一致。当前 normalizer 只校验
   最后一维为 12288，不校验层顺序，也不证明 mean 折叠数值等价；这一步不通过，
   后续权重加载、单步和收敛测试都没有意义。
2. **环境**：确认 PyTorch/torch_npu 配套，`SPECFORGE_DEVICE=npu` 后 backend 是
   HCCL；`--plan` 显示 8 trainer ranks、EP=8。
3. **权重审计**：先运行 `scripts/audit_deepseek_v4_dspark_checkpoint.py`；
   要求 4705 tensors、集合/形状零差异、FP4/FP8 sample exact。然后每个
   rank 记录 local expert 范围与 loaded tensor 数。
4. **单步**：4 anchors、1 objective chunk，完成 forward/backward/step，loss 与
   所有 grad finite。
5. **EP 一致性**：8 ranks 的 replicated parameter checksum 在 step0 和 step2
   一致；local expert checksum 只在 owner rank 变化。
6. **checkpoint**：step1 保存、原拓扑 resume 到 step2；合并后的 state 包含
   3 x 256 experts，且无 missing/unexpected key。
7. **资源**：记录单 rank NPU peak memory、host optimizer memory、step time、HCCL
   time；必要时比较 EP8/DP1 与 EP4/DP2。
8. **收敛 smoke**：固定 32–128 条样本跑至少 100 steps，CE/L1/confidence 都下降，
   `tau_probabilistic` 不退化。
9. **真实训练**：再扩大数据，并沿用 0811 文档里的 gradient spike observe/on
   策略；不要从 2-step smoke 直接推断长跑稳定。

## 10. 已知限制与下一步

1. portable attention/MoE 是 correctness-first 实现，没有融合 NPU kernel；真机
   性能可能不够，下一步可在保持 state contract 不变的前提下替换 RMSNorm、grouped
   linear、MoE 和 attention 边界。
2. FP4/FP8 先解成 BF16，显存高于推理；尚未做 QAT、FP8 training 或重计算专项
   优化。
3. online/disaggregated EP 明确禁止；本轮先把离线 NPU 微调链路做实。
4. 本机无法验证 SGLang V4 capture hook 的真实 tensor layout；normalizer 同时支持
   平均前和平均后形状，但正确 shape 不等于数值正确，必须执行第 9.2 项。
5. 独立 HF draft export 已有 checkpoint 合并能力；生产 fused overlay、重新量化和
   vLLM-Ascend/SGLang 微调权重加载尚未交付。
6. 官方 converter 生成的 MP>1 runtime shard 还带 attention/head tensor-parallel
   切片，不能当作完整权重直接 warm start；当前配方使用官方 HF fused
   checkpoint。
7. Flash config 已提供；Pro 的尺寸（hidden 7168、384 experts、top-6、target
   58/59/60、Markov rank 512）可由同一代码承载，但没有附 Pro recipe，也没有做
   资源预算或真机验证。

因此，对本分支最准确的状态描述是：**DeepSeek-V4 Flash DSpark MoE 的官方结构、
权重初始化、SpecForge 微调目标、Ascend/HCCL EP 拓扑和 checkpoint 生命周期已经
落到代码并通过无设备数学测试；最后一公里是按清单完成 NPU 穿刺、性能优化与 serving
overlay。**
