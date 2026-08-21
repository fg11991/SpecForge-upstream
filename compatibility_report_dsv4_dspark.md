# DeepSeek-V4-Flash-DSpark 微调兼容性核验报告

- 日期：2026-08-13
- 本地分支：`dsv4-dspark-moe` @ `b3c123a`（核验在 `d01e6c2` 上开始，`b3c123a` 落地后已全量复跑，结论未变）
- 远端：`deepseek-ai/DeepSeek-V4-Flash-DSpark`，revision `62af8fffb2f7030cac4de2f0169f5b8d1101b646`，
  `lastModified 2026-07-04T03:15:12Z`
- **等价 repo**：`deepseek-ai/DeepSeek-V4-Flash-0731`（revision `7872f01b…`）与上面那个
  **是同一个模型**，本报告全部结论对它同样成立。实测：全量张量名集合完全相同（72317），
  `mtp.*` 集合相同（4705），所在 shard 相同（46/47/48），shard 48 的 dtype+shape 逐项一致；
  `config.json` 的 `num_hidden_layers` / `dspark_*` / `compress_ratios` / `expert_dtype`
  也逐项一致。**不要因为名字里没有 `-DSpark` 就以为它不含 drafter。**
- 方法：纯静态。只读 HF API 元数据、`*.json`、`inference/*.py`、`encoding/*.py`，
  以及 3 个 shard 的 **safetensors header（HTTP Range）**。
- **实际下载的 tensor 字节数：0。** header 总计 0.50 MB。
  核验前后 `~/.cache/huggingface` 均为 3.4 GB（未变化，且本次全部走 `/tmp/dspark_probe/` 不入 cache）。

---

## 0. 结论先行

**draft 侧完全对得上，可以开箱微调；卡住你的是 target 侧的 key 命名。**

- `DeepseekV4DSparkDraftModel` 的 `state_dict` 与官方 `mtp.*` 张量清单
  **2376 vs 2376，缺失 0、多余 0、真实 shape 不匹配 0**，
  `load_official_checkpoint` 的命名解析 **2376/2376 全部命中**，2329 个 scale 全部找到。
- 逐条比对官方 `inference/model.py` + `inference/kernel.py` + `inference/convert.py` 后，
  mHC、Sinkhorn、attention、grouped `wo_a`、FP4 nibble 顺序、gate、Markov/confidence head
  的语义**全部一致**，包括之前只有注释背书的四个高风险点。
- **1 个阻塞项**：`examples/configs/deepseek-v4-flash-dspark-offline-npu.yaml` 里
  `embedding_key` / `lm_head_key` 用的是 HF 风格名，而这个 repo **整仓都是 runtime 命名**
  （`embed.weight` / `head.weight`，全仓 `model.` 前缀出现 0 次）。
- **1 个待确认项**：offline 抓取端的 `target_last_hidden_states` 必须是**最终 RMSNorm 之后**的
  hidden，不是"hc_head 之后"就够。当前文档措辞有歧义。

---

## 1. DSpark draft 架构与参数量（静态推断）

| 项 | 值 | 依据 |
| --- | --- | --- |
| stage 数 | 3（`mtp.0/1/2`） | 张量清单；`inference/config.json:n_mtp_layers=3` |
| 统一层号 | 43 / 44 / 45 | `DSparkBlock.__init__: layer_id = args.n_layers + stage_id` |
| 并行 block | 5 token | `dspark_block_size=5` |
| mHC streams | 4 | `hc_mult=4`，`mix_hc=(2+4)*4=24`，`hc_dim=4*4096=16384` |
| target 采集层 | 40 / 41 / 42 | `dspark_target_layer_ids` |
| attention | 共享 latent KV（1 head）、64 heads、head_dim 512、partial RoPE 64、SWA 128、learned sink | `compress_ratios[43:46]==[0,0,0]` |
| MoE | 每 stage 256 routed + 1 shared，top-6，inter 2048 | — |
| router | `sqrt(softplus)` + correction bias 选择，**无分组 top-k**，route_scale 1.5 | `Gate.forward` |
| Markov / confidence | rank 256；confidence 输入 4096+256=4352 | 实测 shape `[1,4352]` |

**张量清单（4705 = 2376 权重 + 2329 scale）**

| dtype | 数量 | 内容 |
| --- | ---: | --- |
| `I8`（打包 FP4） | 2304 | 768 routed expert × 3 stage 的 w1/w2/w3 |
| `F8_E8M0` | 2329 | 全部 block scale |
| `F8_E4M3` | 25 | wq_a/wq_b/wkv/wo_a/wo_b ×3、shared_experts ×9、main_proj ×1 |
| `BF16` | 20 | 各 RMSNorm、`gate.weight`、`markov_w1/w2`、`confidence_head.proj` |
| `F32` | 27 | `hc_{attn,ffn}_{fn,base,scale}` ×3、`hc_head_*`、`attn_sink` ×3、`gate.bias` ×3 |

**体积**

- 磁盘上（量化态）：**10.863 GB**
- 反量化成 BF16 可训练参数：**19.846 B 参数 = 39.69 GB**
- EP=8 时每 rank：**2.481 B 参数 = 4.96 GB**（BF16 权重，不含梯度/激活）

---

## 2. 期望 vs 真实 —— 完整 diff

```
expected state_dict entries : 2376
real non-scale tensors      : 2376
real .scale tensors         : 2329

MISSING  (模型要、ckpt 没有) : 0
UNEXPECTED (ckpt 有、模型没有): 0
shape differences           : 2304 处，其中真实不匹配 0 处
                              （2304 处全部是 FP4 nibble 打包：
                               expected (2048,4096) vs ckpt (2048,2048) I8）

load_official_checkpoint 命名解析: 2376/2376 命中
  带 scale     : 2329
  不带 scale   : 47（norm / hc_* / attn_sink / gate.* / markov / confidence）
```

**命名事实（重要）**：这个 repo **整仓使用 runtime 命名**，不是 HF transformers 命名。

```
prefix "model."          出现 0 次
"self_attn"              出现 0 次
"weight_scale_inv"       出现 0 次
"e_score_correction_bias" 出现 0 次
实际形态：embed.weight / head.weight / norm.weight
          layers.0.attn.attn_sink / layers.0.ffn.gate.tid2eid
          mtp.0.attn.wq_a.weight + mtp.0.attn.wq_a.scale
          mtp.0.ffn.gate.bias
```

`load_official_checkpoint` 里 `checkpoint_name = name if name in weight_map else hf_name`
的 **runtime 分支被命中，HF 回退分支对本 repo 是死代码**。这不是 bug，
但开发文档"识别 HF 原始命名与 runtime 命名"这句里的前半句在本 repo 上不会被执行。

**shard 分布（对下载策略很关键）**：4705 个 `mtp.*` 张量全部落在
`model-000{46,47,48}-of-00048.safetensors`，而且这 3 个 shard **只含 `mtp.*`，非 mtp 张量为 0**。

---

## 3. 重点检查项逐条结论

### 3.1 `main_proj` 输入维度 vs `hidden_size × len(target_layer_ids)` — **通过**

真实 `mtp.0.main_proj.weight` = `F8_E4M3 [4096, 12288]`，
scale `[32, 96]`（=4096/128 × 12288/128，128×128 block）。
本地 `Linear(hidden_size * len(target_layer_ids), hidden_size)` = `Linear(12288, 4096)`，
weight `[4096, 12288]`。**逐位对上**。

官方 `DSparkBlock.__init__` 同式：`Linear(args.dim * len(args.dspark_target_layer_ids), args.dim)`。

### 3.2 `target_layer_ids` off-by-one 与 hybrid attention 布局 — **通过**

官方 `Transformer.forward`：

```python
for i, layer in enumerate(self.layers):
    h = layer(h, start_pos, input_ids)
    if i in self.target_layer_ids:
        main_hiddens.append(h.mean(dim=2))       # 对 hc 维求平均
main_hidden = torch.cat(main_hiddens, dim=-1)     # 沿 hidden 维拼接
```

- 取的是 **block i 的输出**（0-based，i=40/41/42），不是输入，不存在 off-by-one。
- **`h.mean(dim=2)` 折叠 mHC streams 是官方行为**，不是本地实现的近似。
  本地 `_normalize_hidden_states` 对 rank-4 做 `mean(dim=-2)`、rank-5 做
  `mean(dim=-2).flatten(-2)`，轴与拼接顺序都一致。**这一条之前只有注释背书，现在有一手证据。**
- hybrid 布局：`compress_ratios` 共 46 项 = 43 target + 3 DSpark。
  索引 0/1 = 0，索引 2..42 交替 4/128（CSA 4 带 indexer、HCA 128 不带），
  索引 43/44/45 = 0 → **DSpark 段是纯滑窗、不压缩、无 indexer**。
  实测：147 个 `indexer` 张量全在 `layers.*`，`mtp.*` 里一个都没有。

### 3.3 confidence head 接线 — **通过**

官方 `forward_head`：

```python
x = self.hc_head(...)                      # post-hc_head, PRE-norm
logits = self.head(self.norm(x), full_logits=True)
... markov bias 逐位加到 logits 上 ...
confidence = self.confidence_head(x, markov_embed)   # 用的是 pre-norm 的 x
```

本地 `prepare_objective_hidden = mtp[-1].norm` 只作用于走 lm_head 的那一路，
`predict_confidence` 吃的是 `forward()` 的返回值（pre-norm）。**两路输入分叉正确。**
真实 `confidence_head.proj.weight` = `BF16 [1, 4352]` = 4096+256，本地 `Linear(4352, 1, bias=False)` 一致。

官方注释另有一句直接印证本地实现：
> proj in the checkpoint is stored in bf16, while the parameter here is stored in fp32

即 checkpoint 本来就是 bf16，inference 侧才升 fp32。本地"存 bf16、forward 里 `.float()`"与之等价。

### 3.4 vocab / vocab mapping / tie_word_embeddings — **通过**

- `vocab_size=129280`、`tie_word_embeddings=false`，真实 `embed.weight` 与 `head.weight`
  都是 `BF16 [129280, 4096]`，是两份独立权重。
- `markov_w1.weight` `BF16 [129280, 256]`（Embedding），`markov_w2.weight` `BF16 [129280, 256]`（Linear 转置形）。
  本地 `nn.Embedding(vocab, 256)` / `nn.Linear(256, draft_vocab)` 在 `draft_vocab == vocab` 时
  shape 完全一致。
- `mtp.*` 命名空间下**没有** embed / head 张量：官方 `Transformer.__init__` 里
  `self.mtp[-1].embed = self.embed; self.mtp[-1].head = self.head`，
  `convert.py` 也显式 `if name.startswith("mtp.") and ("emb" in name or name.endswith("head.weight")): continue`。
  → **DSpark 复用 target 的 embedding 和 lm_head**，与本地"冻结 target 表、只训 `mtp.*`"一致。
- 剪枝路径（`use_draft_vocab`）本 recipe 未启用（`draft_vocab_size == vocab_size`），
  `markov_w2` 的 d2t 行选分支不会触发。

### 3.5 `num_speculative_tokens=7` — **不适用（你记错了）**

本 repo 没有 7 这个数。官方是 `dspark_block_size = 5`（一次出 5 个并行草稿 token）
× 3 个 DSpark stage。`config.json` 与 `inference/config.json` 双向印证，
`get_dspark_topk_idxs(window_size, bsz, block_size=5, start_pos)` 也是 5。
7 大概来自别的模型（或 EAGLE 系的 tree depth）。本地配置写的是 5，**正确**。

### 3.6 stage 数：`num_nextn_predict_layers=1` vs `n_mtp_layers=3` — **风险（当前蒙对）**

- HF `config.json`：`num_nextn_predict_layers: 1`
- `inference/config.json`：`n_mtp_layers: 3`
- 真实张量：`mtp.0/1/2` **确实是 3 个 stage**

本地 `_draft_config_from_dict` 用 `payload.setdefault("dspark_num_layers", 3)` 硬编码 3，
读的是 HF config，**恰好蒙对**。我实跑过转换：

```
converted architectures : ['DeepseekV4DSparkDraftModel']
converted num_layers    : 3      <- 来自硬编码，不是来自 config
converted target_layers : [40, 41, 42]
converted block_size    : 5
```

Flash 没问题，但 Pro 或后续版本一旦 stage 数不是 3，这里会静默建错模型。
建议改成优先读 `n_mtp_layers`，其次按 `len(compress_ratios) - num_hidden_layers` 推导，
都拿不到再回退 3。

### 3.7 feature key 契约 — **通过（draft 侧），风险（抓取侧）**

本地 offline 契约是 `input_ids` / `loss_mask` / `hidden_states` / `target_last_hidden_states`，
`build_dspark_offline_reader` 的 `feature_keys` 与 normalizer、collator 三处一致，
`hidden_states`（复数）不存在单数 `hidden_state` 的用法。draft 侧无 KeyError 风险。

**但抓取侧有一条必须钉死**（见 §3.9）。

### 3.8 target 是 MoE + FP8/FP4：抓隐状态的假设 — **通过（无需 dequant）**

- 要抓的两样东西都是 **BF16 激活**，不是权重，与 target 用 FP8/FP4 存权重无关，
  抓取端不需要任何 dequant。
- draft 自身**确实含 MoE**：每 stage 256 routed expert，占 draft 全部参数的
  19.33 B / 19.85 B ≈ **97.4%**。所以 EP 切分是必需的，不是可选优化。
- 本地 EP 方案（replicated token + 本 rank 局部 expert + all-reduce + shared expert 单独加）
  与官方 `MoE.forward` 的结构一致：官方也是
  `y[idx] += expert(...)` → `if world_size>1: dist.all_reduce(y)` → `y += shared_experts(x)`，
  shared expert 在 all-reduce **之后**加，只加一次。本地实现同序。

### 3.9 【风险】`target_last_hidden_states` 必须是 **最终 norm 之后**

官方教师分布是：

```python
h = layer.hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
logits = self.head(self.norm(h))          # 注意 self.norm
```

而本地 `apply_teacher_head` 直接 `lm_head(hidden_states)`，**不加任何 norm**
（这是 `d01e6c2` 之前那次修复刻意分出来的一路）。

因此抓取端写出的 `target_last_hidden_states` 必须已经是 **`norm(hc_head(h))`**。
开发文档现在的说法是"必须是 target 全局 `hc_head` 之后的 rank-2/3 tensor"——
**这句话在 norm 这一层是有歧义的**。如果抓成 hc_head 之后、norm 之前，
教师分布会整体错掉，而且形状全对、训练照跑、loss 照降，不会报错。

**建议**：文档改成"必须是最终 RMSNorm（`model.norm`）之后、送入 lm_head 之前的 hidden"，
并在上机第一步用同一条 prompt 对拍：`lm_head(captured) == 官方 logits`。

### 3.10 【阻塞】target embedding / lm_head 的 key 名不对

```
model.embed_tokens.weight   present=False      <- 你 yaml 里写的
lm_head.weight              present=False      <- 你 yaml 里写的
embed.weight                present=True       <- 真实
head.weight                 present=True       <- 真实
```

`examples/configs/deepseek-v4-flash-dspark-offline-npu.yaml:10-11`：

```yaml
embedding_key: model.embed_tokens.weight
lm_head_key: lm_head.weight
```

`TargetEmbeddingsAndHead._load_weights` 会走到

```python
missing_from_index = sorted(set(required_keys) - weight_map.keys())
if missing_from_index:
    raise ValueError("Required target weight keys are missing from the checkpoint index: ...")
```

**直接 raise**。DSpark 的 CE / 分布 L1 / confidence 三个目标都依赖冻结的 target
embedding + lm_head，这一条不修，offline 训练起不来。

改成：

```yaml
embedding_key: embed.weight
lm_head_key: head.weight
```

顺带确认：`load_target_config` 对这个 repo 是**可用的**。`AutoConfig.from_pretrained` 会失败
（`model_type: deepseek_v4` transformers 不认，且仓里没有 `modeling_*.py` / `auto_map`），
但它会回退到裸 `config.json`，我实跑验证过：

```
type: _RawConfigShim
  vocab_size = 129280   hidden_size = 4096
  tie_word_embeddings = False   num_hidden_layers = 43
TargetEmbeddingsAndHead(config) OK: [129280, 4096] [129280, 4096]
```

（附带结论：yaml 里的 `trust_remote_code: true` 对这个 repo 是**空操作**，
仓库里没有任何远程代码。留着无害。）

### 3.11 量化语义逐条核对 — **全部通过**

官方 `convert.py`：

```python
FP4_TABLE = [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0, 0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0]
low  = x & 0x0F
high = (x >> 4) & 0x0F
x = torch.stack([FP4_TABLE[low], FP4_TABLE[high]], dim=-1).flatten(2)
```

本地 `_FP4_TABLE` 与之**逐元素相同**，`torch.stack((table[packed & 0x0F], table[(packed>>4)&0x0F]), dim=-1).flatten(-2)`
与之**逐字相同**，包括 **low nibble 在前**。

- FP4 block scale：每 32 个 K 元素一个 → 实测 `w1.scale [2048,128]`，128×32=4096 ✓
  本地 `repeat_interleave(32, dim=-1)` ✓
- FP8 block scale：128×128 → 实测 `wo_a.scale [64,32]` = 8192/128 × 4096/128 ✓
  官方 `weight.unflatten(0,(-1,128)).unflatten(-1,(-1,128)) * scale[:,None,:,None]`
  与本地 `repeat_interleave(128,-2).repeat_interleave(128,-1)` **数学等价** ✓
- E8M0：`scale_fmt: ue8m0`，本地 `2^(raw-127)`、255→NaN ✓
- dtype 支持：本机 torch 2.12 三个都在
  （`float8_e4m3fn` / `float8_e8m0fnu` / `float4_e2m1fn_x2` 全部 present）。
  **上机前需在 torch_npu 环境里复验这三个 dtype 是否存在**——
  `_decode_scale` 用 `getattr(torch,"float8_e8m0fnu",None)`，缺失时会静默走
  `scale.float()` 分支，产出错误数值而不报错。

**用真实 shape 跑的反量化实测**：

```
FP4 expert : packed (2048,2048) + scale (2048,128) -> (2048,4096)  模型要 (2048,4096)  OK
FP8 wo_a   : weight (8192,4096) + scale (64,32)    -> (8192,4096)  模型要 (8192,4096)  OK
```

### 3.12 前向语义逐条核对 — **全部通过**

| 项 | 官方 | 本地 | 结论 |
| --- | --- | --- | --- |
| mHC `pre` | `sigmoid(mixes[j]*scale[0]+base[j])+eps` | 同 | 一致 |
| mHC `post` | `2*sigmoid(mixes[j+hc]*scale[1]+base[j+hc])` | 同 | 一致 |
| mHC `comb` 索引 | `mixes[j*hc+k+2hc]` | `mixes[...,2hc:].view(hc,hc)[j,k]` | 一致 |
| Sinkhorn | softmax(-1)+eps → 列归一 → (iters-1)×[行,列] | 同 | 一致（含列优先与 iters-1） |
| `hc_post` 收缩轴 | `sum(comb.unsqueeze(-1)*residual.unsqueeze(-2), dim=2)` | 同 | 一致（防住了 transpose bug） |
| `hc_pre` 归一化位置 | `F.linear(x, fn) * rsqrt` | 同 | 一致 |
| q 归一化 | `wq_b(q_norm(wq_a(x)))` 后再 RMS（无权重） | 同 | 一致 |
| RoPE | 仅作用于末 64 维 | 同 | 一致 |
| 输出侧 inverse RoPE | `apply_rotary_emb(o[...,-rd:], freqs, True)` | `inverse=True` | **一致（此前无证据）** |
| `wo_a` 分组布局 | `weight.view(n_groups, o_lora_rank, -1)` + `einsum("bsgd,grd->bsgr")` | `view(groups, out_per_group, in)` + `bmm` | **一致（此前无证据）** |
| YaRN | `compress_ratio==0` 时禁用、用 base `rope_theta` | `rope_scaling=None` | **一致（官方注释原文印证）** |
| softmax scale | `head_dim ** -0.5` | 同 | 一致 |
| attention sink | 参与 softmax 分母、不进分子 | 同 | 一致 |
| gate 选择 | `(scores+bias).topk(6)`，**无分组 top-k** | 同 | 一致 |
| gate 权重 | `original_scores.gather` 后归一 × 1.5 | 同 | 一致 |
| hash routing | 仅 `layer_id < 3`，DSpark 是 43-45 → 不走 | 本地无 hash | 一致 |
| Expert | fp32 中间、`clamp(gate,max=L)`/`clamp(up,±L)` | 同 | 一致 |
| MoE 加法序 | all_reduce(routed) 后再加 shared | 同 | 一致 |
| noise block | slot0=真实 anchor token，其余=128799 | 同 | 一致 |
| draft 位置 | context 末位 +1 起共 5 个 | `draft_position_offset=1` | 一致 |
| block 内可见性 | 全 5 个并行可见（非因果） | `parallel_draft_visibility` | 一致 |
| anchor 是否进 context | 进（写入 kv_cache 后再 concat） | `include_anchor_context=True` | 一致 |
| Markov bias 时机 | lm_head 之后逐位加 | 同 | 一致 |

### 3.13 【风险】F32 checkpoint 张量被降到 BF16

`d01e6c2` 删掉 `_apply` 之后，本地所有参数统一 BF16 存储。对照真实 dtype：

- 各 RMSNorm、`gate.weight`、`markov_w1/w2`、`confidence_head.proj` —— checkpoint 本来就是 **BF16**，
  官方注释也明说"stored in bf16"。本地无损失。**之前我担心的这部分是多虑了。**
- 但 **27 个张量在 checkpoint 里是 F32**：`hc_{attn,ffn}_{fn,base,scale}`、`hc_head_*`、
  `attn_sink`、`gate.bias`。其中 `gate.bias` 现在是 F32 buffer（无损），
  **其余 24 个会在载入时被 BF16 舍入**。

`hc_*_scale ≈ 1`、`hc_*_base ≈ 0` 量级上 BF16 相对误差约 0.4%，而 `comb` 之后还要过
20 次 Sinkhorn 迭代。这是**唯一一处"起点不等于官方 drafter"的地方**，需要实测量化影响。

---

## 4. 明确结论：开箱能否微调

**能，但要先改 2 行 yaml。** 按阻塞程度排序：

| # | 级别 | 问题 | 改动 |
| --- | --- | --- | --- |
| 1 | **阻塞** | `embedding_key` / `lm_head_key` 用了 HF 命名，真实 repo 是 runtime 命名 | yaml 改 `embed.weight` / `head.weight` |
| 2 | ~~阻塞~~ **已修复** | 若把 `draft_model_config` 指向官方 config.json，schema 的 EP 校验会拒 | `b3c123a` 已抽出 `specforge/config/draft_config.py:is_deepseek_v4_dspark_draft_config`，schema 与 loader 复用。实测官方形态 config（`architectures=["DeepseekV4ForCausalLM"]`）现已 ACCEPTED |
| 3 | **风险** | `target_last_hidden_states` 是否为最终 norm 之后，文档措辞有歧义 | 文档改写 + 上机第一步用 `lm_head(captured)` 对拍官方 logits |
| 4 | **风险** | 24 个 F32 的 hc_* / attn_sink 被降到 BF16 | 上机对拍一次 drafter 输出；若不可忽略再考虑单独隔离这几个张量 |
| 5 | **风险** | `torch.float8_e8m0fnu` 在 torch_npu 上若缺失，`_decode_scale` 会静默走错分支 | 上机前 3 行断言；缺失就 raise，不要静默 |
| 6 | **风险** | 抓取端能否拿到 layer 40/41/42 的 mHC streams 并做 `mean(dim=2)` —— 仓里无 transformers 建模代码，全靠 sglang | 只能实跑；这是整条链路唯一无法静态核验的部分 |
| 7 | 提示 | `_draft_config_from_dict` 硬编码 `dspark_num_layers=3` | Flash 正确；Pro 之前改成读 `n_mtp_layers` |
| 8 | 提示 | `num_speculative_tokens=7` 不存在于本模型 | 无需改动，本地已是 5 |
| 9 | 提示 | `trust_remote_code: true` 是空操作 | 可留 |

除此之外，**架构、命名、shape、量化语义、前向数学五个层面全部核验通过，没有需要修改的适配点。**

---

## 5. draft-only 权重体积与冒烟建议

| 项 | 值 |
| --- | --- |
| `mtp.*` 张量数 | 4705（2376 权重 + 2329 scale） |
| 所在 shard | `model-000{46,47,48}-of-00048.safetensors` |
| **这 3 个 shard 的纯度** | **100%，非 `mtp.*` 张量为 0** |
| 磁盘体积（量化态） | **10.863 GB** |
| 反量化后 BF16 | 19.846 B 参数 / 39.69 GB |
| EP=8 时每 rank | 2.481 B / 4.96 GB（仅权重） |

**建议做这个冒烟测试，性价比很高。** 最小下载集：

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --revision 62af8fffb2f7030cac4de2f0169f5b8d1101b646 \
  --include "config.json" "model.safetensors.index.json" \
            "model-00046-of-00048.safetensors" \
            "model-00047-of-00048.safetensors" \
            "model-00048-of-00048.safetensors" \
  --local-dir ./ckpt/dsv4-flash-dspark-draft-only
```

约 **10.9 GB**（占全仓 166.9 GB 的 6.5%），能覆盖：

- `load_official_checkpoint` 端到端真实跑通（本报告只验证了名字与形状能对上，
  **没有验证任何一个真实字节的数值**）
- FP4 / FP8 反量化的**数值**正确性（把反量化结果和 `convert.py`
  的 `cast_e2m1fn_to_e4m3fn` 输出逐元素对比）
- 一次真实权重下的 tiny forward，确认 loss 不是 NaN
- §3.13 那条：F32 → BF16 舍入对 drafter 输出的实际影响

注意这 3 个 shard **不含** target 的 `embed.weight` / `head.weight`
（分别在 shard 1 和 45），所以它只够验证 draft 载入，
不够跑完整的 DSpark 目标函数。要跑完整目标还需另加 shard 1（1.06 GB）和 45（1.06 GB）。

---

## 6. 三个常见疑问

### 6.1 「DSv4 的 DSpark 真的是 MoE 吗？」—— 是，无可争议

不是推断，是数出来的：

```
routed expert 权重张量 : 2304  = 256 experts × 3 stage × (w1/w2/w3)
每 stage 的 expert id  : 0 - 255
gate 张量              : mtp.{0,1,2}.ffn.gate.weight  BF16 [256, 4096]
                         mtp.{0,1,2}.ffn.gate.bias    F32  [256]
shared expert 张量     : 9  = 3 stage × (w1/w2/w3)
draft 总参数 19.846 B，其中 ffn(MoE) 19.406 B = 97.8%
```

官方 `inference/model.py` 的类继承也直说了：`class DSparkBlock(Block)`，
而 `Block.__init__` 里 `self.ffn = MoE(layer_id, args)` —— DSpark stage 就是一个完整的 V4 block，
只是把 attention 换成 `DSparkAttention` 并挂上 Markov/confidence head。**没有 dense 变体。**

换句话说：**MoE 就是这个 drafter 的主体**（97.8% 的参数），
不是可选组件。任何声称"不需要 MoE 也能微调它"的说法，都不是在微调这份权重。

### 6.2 「有人说用 GLM 的 5 层 dense config 改改也能微调」—— 他说的是另一件事

两句话都对，但指的不是同一个任务。本仓库里同时存在两条 DSpark 路径：

| | 官方 DSpark 微调（本分支新增） | GLM 式 dense drafter（仓库原有） |
| --- | --- | --- |
| 注册类 | `DeepseekV4DSparkDraftModel` | `DSparkDraftModel`（DFlash backbone） |
| 参考配置 | `configs/deepseek-v4-flash-dspark.json` | `configs/glm-5.2-dspark.json` |
| 起点 | 官方 19.85 B 已训练权重 | **随机初始化** |
| 架构 | 3 stage MoE（256 routed + 1 shared）+ mHC + 共享 latent attention | N 层 dense Qwen3 式 transformer |
| 投影器 | `main_proj: Linear(4096×3 → 4096)` | `fc: Linear(hidden×len(target_layer_ids) → hidden)` |
| 参数量 | 19.85 B | 通常 1–3 B |
| 是否需要 EP | 必须（expert 占 97.8%） | 不需要 |
| 能否喂回官方 serving | 能（`mtp.*` 布局一致） | **不能**——vLLM-Ascend / SGLang 的 V4 DSpark 路径要的是 fused `mtp.*` FP4 布局 |
| 达到官方接受率所需数据 | 少（本来就是训好的） | 多得多，要从零学 |

`DSparkDraftModel` 的 `fc = nn.Linear(len(target_layer_ids) * hidden_size, hidden_size)`
和 V4 的 `main_proj` 是同一个概念，所以它**确实**可以对着 DeepSeek-V4 的 target 训练，
需要改的只有几行：`hidden_size: 4096`、`vocab_size: 129280`、`num_target_layers: 43`、
`dflash_config.target_layer_ids`、`mask_token_id: 128799`。"稍微改一下"这个描述是准确的。

**但那是训一个全新的替代 drafter，不是微调 DeepSeek 发布的那个。** 两条路线的取舍：

- 想要**低成本、快速拿到一个能用的 draft**，且不介意重新训、也不打算接官方 serving
  → 走 dense 路线，不需要本分支的任何东西。
- 想要**站在官方已训练好的 drafter 上继续微调**（省数据、直接继承其接受率，
  微调后还能覆盖回原 fused shard 接官方推理）
  → 必须是 MoE，也就是本分支。

顺带澄清一个数字：`num_speculative_tokens=7` 的来源是
`configs/glm-5.2-dspark.json` 里的 `"block_size": 7`。那是 GLM-5.2 recipe 的并行草稿长度，
DeepSeek-V4 官方是 **5**（`config.json` 与 `inference/config.json` 双向印证）。
两个模型本来就不同，不是谁记错。

### 6.3 「现在的 MoE 训练框架已经可以微调他们的模型了吗？」—— 分三层说

**架构层：可以，且已逐项核验。** 见 §2、§3.11、§3.12——命名、shape、量化语义、
前向数学四个层面对官方参考实现全部一致，没有需要修改的适配点。

**配置层：还差 1 处。** §4 的第 1 项——yaml 的 `embedding_key`/`lm_head_key`。
（第 2 项 schema 校验已在 `b3c123a` 修复并实测通过。）

**运行层：尚无任何证据。** 必须明确：本报告**没有验证过任何一个真实字节**，
也没有在 NPU / HCCL / 多 rank 上跑过。仍然未知的是——

1. 真实权重载入与反量化的**数值**正确性（§5 的 10.9 GB 冒烟能覆盖）
2. offline 抓取端能否拿到 layer 40/41/42 的 mHC streams 并做 `mean(dim=2)`，
   以及 `target_last_hidden_states` 是否取在最终 norm 之后（§3.9）——
   这是整条链路唯一**无法静态核验**的部分，仓库里没有 transformers 建模代码，全靠 sglang
3. `torch.float8_e8m0fnu` 在 torch_npu 上是否存在（§3.11，缺失会静默出错）
4. NPU 显存、吞吐、HCCL 建组

所以准确的说法是：**"能微调"这件事在架构层已经被证明；"现在就能跑通"还没有。**
建议顺序：改 §4 的两项 → 拉 10.9 GB 做载入冒烟 → 抓取端数值对拍 → 单机 8 卡单步 → 扩多机。

---

## 附：核验产物

- `/tmp/dspark_probe/real_mtp_tensors.json` —— 4705 个张量的 name / dtype / shape / data_offsets
- `/tmp/dspark_probe/fetch_headers.py` —— header-only Range 拉取脚本
- `/tmp/dspark_probe/diff_state.py` —— meta-device 期望清单与 diff
- `/tmp/dspark_probe/inference_{model,convert,kernel,generate}.py` —— 官方参考实现快照

下载量核账：header 0.50 MB + `config.json` / `index.json` / `inference/*.py` / `encoding/*.py`
约 12 MB，**tensor 字节 0**。
