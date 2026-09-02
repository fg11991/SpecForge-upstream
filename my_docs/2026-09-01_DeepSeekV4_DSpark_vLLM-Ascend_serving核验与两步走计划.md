# DeepSeek-V4 DSpark 微调权重回灌 vLLM-Ascend：上机方案

- 日期：2026-09-01
- 分支：`dsv4-moe-dspark-trainable`
- 核验对象：
  - `fg11991/vllm-ascend`，分支 `vllm-0.27-dsv4`，commit `dedbb34`
  - `fg11991/vllm`，分支 `vllm-0.27-dsv4`
- 方法：静态读两边源码，**加上 A2 容器与共享盘上的五条实测**（§1 逐条标注）。
- **本文取代同日初版（`68c4b64`）**。初版是纯静态推断，其中四节已被实测推翻或收窄：
  description key 的担忧（不成立）、导出件疑似缺 expert（不成立）、
  独立 draft 目录没有先例所以排后（应该排第一）、以及只提 `architectures` 而漏了
  `model_type`（后者才是硬性的，见 §2.2）。旧版在 git 历史里。

---

## 0. 结论先行

1. **走独立 draft 目录**。`load_dspark_model` 用
   `get_model(vllm_config=draft_vllm_config, model_config=draft_model_config)`
   建 draft，权重从 draft 自己的路径加载——这条路是支持的，不用碰 target 目录。
2. **导出件是完整的**，但 `config.json` 必须改三个字段，其中 `model_type` 是硬性的
   ——vLLM 那段自动覆写发生在 `ModelConfig` 构造之后，救不了构造阶段的解析失败（§2.2，实测）。
3. **ModelSlim warm start 是对的**：description 的 key 就是磁盘张量名（实测 7022/7022）。
4. **首次上机跑通了，但接受率 4.6%，官方基线是 88%**（§5）。特征契约已逐行排除
   （§4.1），嫌疑集中在 warm start 是否真的生效、以及 100 步训练是否把它训坏了。
5. 次要风险已定性：声明为 FP32 的 27 个参数里，24 个在**训练全程**就是 BF16
   （FSDP1 的 flat-param 约束所致），但运算全程上采、`gate.bias` 已被单独保住，serving 无影响（§4.2）。

---

## 1. 已确认的事实

| # | 结论 | 证据 | 类型 |
| --- | --- | --- | --- |
| 1 | description 的 key **就是**磁盘张量名 | `mtp.*` 条目 7022，与 index 的 `mtp.*` 张量名交集 7022 | 实测 |
| 2 | key 风格是原生 `attn`/`ffn`/`w1`，不是 `self_attn`/`mlp`/`gate_proj` | `attn style = 42`（= 3 stage × 14），`self_attn style = 0` | 实测 |
| 3 | 导出件完整 | 2378 张量、2304 expert、38.9 GiB（= 41.8 GB）、`mtp.0.embed` 与 `mtp.2.head` 都在 | 实测 |
| 4 | same-checkpoint DSpark 服务本来就能跑起来 | 用户在 A2 八卡上以 `--quantization ascend` + `{"method":"dspark","num_speculative_tokens":7}` 拉起过 | 实测 |
| 5 | `/dpc/hot/x00873006/...` 与 `/sharenfs/...` 是同一份 checkpoint | 用户确认 | 实测 |
| 6 | 独立 draft 目录被支持 | `vllm/v1/worker/gpu/spec_decode/dspark/utils.py::load_dspark_model` 用 `model_config=draft_model_config` 调 `get_model` | 源码 |
| 7 | draft 的 `architectures` / `model_type` 由 vLLM 强制覆写，但**太晚**——`ModelConfig` 先构造先失败 | `vllm/config/speculative.py:954-963`；实测报错见 §2.2 | 源码+实测 |
| 8 | 独立路径下 draft 不继承 target 量化 | `speculative.py:703` 的 `self.model is None` 分支不触发 → `self.quantization` 保持 None；`patch_dspark.py` 的 `draft_model_config.model == model_config.model` 也不成立 | 源码 |
| 9 | drafter 会保留自己的 embed/head | `spec_decode/eagle/utils.py:12-25` 的 `_should_share`：draft 有自己的且与 target 不相等就不共享 | 源码 |
| 10 | draft 层用 target 的 hf_config | `vllm_ascend/models/deepseek_v4/model.py:656-657` | 源码 |
| 11 | `mtp.0` → layer 43 → `compress_ratio=0` → SWA、theta 10000、不启 yarn | `vllm_ascend/utils.py:87-100,110-115`；`model.py:519-525` | 源码 |
| 12 | 采到的 `last_hidden_states` 是最终 RMSNorm **之后**的 | capture patch 在 `self.norm(hidden_states)` 之后返回，并显式抑制 `hidden_states_before_norm`（patch 文档第 4 条） | 源码 |
| 13 | W8A8 条目数闭合 | 2322 × 3 = 6966 `W8A8_DYNAMIC`，加 56 `FLOAT` = 7022 | 推导+实测 |
| 14 | 训练与 serving 的 aux 特征契约**完全一致**（层 40/41/42 的输出、`mean(dim=1)` 折叠、`main_norm(main_proj)`） | 见 §4.1 的逐项对照 | 源码 |
| 15 | 独立 draft 目录能真正拉起服务 | 2026-09-02 在 A2 八卡上起来了，`--speculative-config` 带 `"model"` | 实测 |
| 16 | 该 drafter 的接受率 pos0 = 4.6%（20/439），pos1-6 全 0 | `/metrics`，见 §5 | 实测 |

补充事实（用户提供）：训练已跑通并导出；warm start 用的就是
`DeepSeek-V4-Flash-0731-w8a8` 里的 `mtp.*`（即 INT8 反量化的起点）；
训练指标正常、梯度不爆、accuracy 有上升；接受率基线待测。

---

## 2. 上机方案：独立 draft 目录

### 2.1 为什么能走通（完整源码链条）

```
speculative_config.model = <draft 路径>          # 用户显式给出，speculative.py:703 分支不触发
  → self.quantization 保持 None                  # 同上，不会被赋成 target 的 "ascend"
  → draft_model_config = ModelConfig(<draft 路径>, quantization=None)
  → speculative.py:954-963 强制 architectures = ["DSparkDraftModel"], model_type = "deepseek_v4"
  → patch_dspark: draft_model_config.model != model_config.model → 不继承 target 量化
  → get_draft_quant_config(vllm_config) → draft 目录里没有 quant_model_description.json
                                          且 config.json 里没有 quantization_config → None
  → load_dspark_model: get_model(draft_vllm_config, model_config=draft_model_config)
  → DSparkDeepseekV4ForCausalLM 以未量化建层，从 draft 目录加载 BF16 权重
```

三个 DSpark 层仍然由 **target 的 hf_config** 构造（`model.py:656-657`），
所以 `compress_ratios` / `num_hash_layers` / `index_topk_*` / `compress_rope_theta`
这些 draft config 里没有的字段自动从 target 来，不用补、不用软链。
tokenizer 同理，走 target 的。

### 2.2 draft 目录该长什么样

**两个文件，就这两个：**

```
deepseek-v4-flash-dspark-export/
├── config.json
└── model.safetensors      # 2378 张量、38.9 GiB
```

> **绝对不要把 target 的 `quant_model_description.json` 软链进来。**
> `vllm_ascend/quantization/utils.py:165` 判定"是不是 ModelSlim 模型"的唯一依据
> 就是这个文件存不存在。链进去等于声明这个 BF16 draft 是量化的，
> 然后拿 target 的标签去查我们的 BF16 权重。

`config.json` 必须改三个字段。用脚本改（它同时会检查目录里没有混进
`quant_model_description.json`，并把训练侧 config 备份成
`config.json.specforge-training`，这样目录还能被 `AutoDraftModel.from_pretrained`
读回去继续微调）：

```bash
python scripts/prepare_dspark_serving_config.py --draft-dir <导出目录>
# 若下一步报 rope 相关的 KeyError，再加 --drop-rope-scaling
```

改的就是这三个字段：

```json
"model_type": "deepseek_v4",
"architectures": ["DSparkDraftModel"],
"n_mtp_layers": 3
```

**`model_type` 是硬性的，不改起不来**（实测）。`deepseek_v4_dspark` 不在
transformers 也不在 vLLM 的配置注册表里（`vllm/transformers_utils/config.py:89`
只有 `deepseek_v4`），`ModelConfig` 在构造阶段就抛：

```
Error parsing config for <draft dir>: The checkpoint you are trying to load has
model type deepseek_v4_dspark but Transformers does not recognize this architecture.
...
pydantic_core.ValidationError: 1 validation error for SpeculativeConfig
```

`speculative.py:954-963` 那段"自动把 `model_type` 设成 `deepseek_v4`、
`architectures` 设成 `DSparkDraftModel`"的代码在 `SpeculativeConfig.__post_init__`
里，**比 `ModelConfig` 构造晚**，救不了这个错。所以这两个字段要在导出侧就写对，
不能指望 vLLM 覆写。

改成 `deepseek_v4` 是安全的：`vllm/transformers_utils/configs/deepseek_v4.py` 的
`DeepseekV4Config` 总共 23 行，只显式接 `max_position_embeddings` / `rope_scaling` /
`rope_parameters` / `rope_theta`，其余全部走 `**kwargs` 变成属性——没有必填字段、
没有校验器，我们的 `dspark_*` 字段原样保留，缺的字段也不会报错（三个 DSpark 层
读的是 target 的 config）。

`n_mtp_layers` 是 `vllm_ascend/models/deepseek_v4/dspark.py:65-69` 的首选字段
（其次 `dspark_num_mtp_layers`，再默认 3）。我们导出的是 `dspark_num_layers`，
靠默认值蒙对，写死更稳。

其余字段保持原样。**若下一步撞上 rope 相关的 KeyError**，删掉整个 `rope_scaling`
字段：`DeepseekV4Config` 取的是 `rope_scaling or rope_parameters`，而我们那个 yarn
dict 里没有 `rope_theta`；DSpark 层不用 yarn，删掉即可让 `rope_parameters` 生效。

### 2.3 启动命令

在你已经跑通的那条上加一个 `"model"`：

```bash
DRAFT=/sharenfs/w00958190/dsv4-dspark/0901_export/deepseek-v4-flash-dspark-export

vllm serve /dpc/hot/x00873006/DeepSeek-V4-Flash-0731-w8a8 \
    --max-model-len 133072 \
    --max-num-batched-tokens 8192 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 32 \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --no-enable-prefix-caching \
    --no-disable-hybrid-kv-cache-manager \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
    --quantization ascend \
    --port 8900 \
    --block-size 128 \
    --speculative-config "{\"method\":\"dspark\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":7,\"enforce_eager\":true}" \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

`--quantization ascend` 只作用于 target 的 ModelConfig；draft 的量化由
`speculative_config.quantization` 决定，不给就是 None。

**显存**：BF16 drafter 在 EP8 下约 5 GB/rank，比 same-checkpoint 的 W8A8 版本多约
2.5 GB/rank。target 权重实测 36.99 GB/rank，卡上原有约 23.8 GB 余量，放得下，
但 `--gpu-memory-utilization 0.9` 下 KV 池会相应变小。

### 2.4 embed / head 的归属

`load_dspark_model` 在建完 draft 后会决定是否让 draft 共享 target 的 embedding/head：

```python
# spec_decode/eagle/utils.py:12-25
def _should_share(eagle, flag, draft, target):
    if not getattr(eagle, flag, False) or draft is None:
        return True            # draft 没有自己的 → 共享 target 的
    if target is None:
        return False
    return torch.equal(draft.weight, target.weight)   # 相等才共享
```

`flag` 由加载时的 `process_eagle_weight` 置位（`vllm/model_executor/models/utils.py:1028-1047`）：
权重名里带 `embed_tokens` / `lm_head` 就置 True。我们的 `mtp.0.embed.weight` 和
`mtp.2.head.weight` 经 `_remap_dspark_name` 正好变成这两个名字，而且数值与 target 的
不相等（compat report：差异分别达 3.7 和 4.4），所以 **drafter 会保留自己的 embed 和 head**
——这正是 `642c237` 想要的行为。

### 2.5 判据

```bash
curl -s localhost:8900/metrics | grep -E "spec_decode_num_drafts|spec_decode_num_accepted_tokens_per_pos"
```

和**同一组 prompt、同样 `num_speculative_tokens`** 下 same-checkpoint 官方 drafter
的数字比。官方 e2e 的 golden 参考值（`tests/e2e/.../test_dspark_deepseekv4.py`，
w4a8 权重、TP4）是 `[0.88, 0.74, 0.58, 0.49, 0.40, 0.30, 0.18]`，
但**不要拿它当基线**——权重、并行度、prompt 都不同。基线必须是你自己那台机器上
官方 drafter 的实测值。

---

## 3. 备用方案：same-checkpoint 覆盖

独立目录走不通时才用。实测的分片布局（`quant_model_weights.safetensors.index.json`）：

| 分片 | mtp 张量 | 非 mtp 张量 |
| --- | ---: | ---: |
| `-00068-of-00074` | 1306 | **185** |
| `-00069-of-00074` | 1080 | 0 |
| `-00070-of-00074` | 1530 | 0 |
| `-00071-of-00074` | 1494 | 0 |
| `-00072-of-00074` | 1530 | 0 |
| `-00073-of-00074` | 82 | **5** |

合计 7022 ✓。**00068 和 00073 是混合分片**，这条路因此比想象中麻烦：

vLLM 用 index 决定加载哪些**文件**，但对选中的文件是遍历里面**所有**张量。
所以这两个文件只要还被 index 引用（它们含非 mtp 张量，必须引用），
里面的旧 INT8 `mtp.*` 就会被一起读出来，和新的 BF16 撞名。
**这两个分片必须物理重写**（读出、剔掉 `mtp.*`、重新 `save_file`），
只改 index 不够。

其余步骤：新目录里软链 target 全部文件 → `rm` 掉要改写的软链（**不 `rm` 直接写会穿透
到原始 294 GB checkpoint**）→ 挂上我们的 BF16 分片 → 重写 index（`mtp.*` 指向新分片）
→ 重写 description（删掉旧的 7022 条 `mtp.*`，按**同样的 key 风格**写入 2378 条 `FLOAT`）。
config.json 用 target 的，我们导出的那个在这条路上完全用不上。

`FLOAT` 标签会让 `is_layer_skipped_ascend`（`modelslim_config.py:834-871`）
返回未量化的 method。这也是 vllm-ascend 自己对 `deepseek_mtp` 的做法
（`modelslim_config.py:176-178` 的 NOTE：msmodelslim 不生成 MTP 层信息，请手工设为 FLOAT）。

---

## 4. 真正的风险清单（按影响排序）

### 4.1 ~~mHC 折叠~~ —— **已排除**（2026-09-02）

初版把这条列为最高风险：capture patch 自己写着 `captured.mean(dim=1)` 这个 mHC
折叠"NOT VERIFIED HERE"。现在逐行对完了，**训练侧和 serving 侧完全一致**：

| | 训练（SGLang capture patch） | serving（vllm-ascend） |
| --- | --- | --- |
| 请求的层号 | `dflash_config.target_layer_ids` = `[40,41,42]` 原样（`model_providers.py:270-281`） | `tuple(i+1 for i in dspark_target_layer_ids)` = `(41,42,43)`（`model_runner_v1.py:675-689`） |
| 捕获时机 | `if i in layers_to_capture` → 层 i 的**输出** | `if layer.layer_idx + 1 in aux_layers` → 层 `layer_idx` 的**输出**（`model.py:879-888`） |
| 实际捕获 | **层 40/41/42 的输出** | **层 40/41/42 的输出** |
| mHC 折叠 | `captured.mean(dim=1)` | `hidden_states.mean(dim=1)`（`model.py:888`） |
| 三层拼接 | `LogitsProcessor` 沿最后一维拼成 `[T, 3H]` | `aux_hidden_states` 列表交给 draft |
| 投影 | `_project_target_hidden`：`main_norm(main_proj(x))`（`:1259-1260`） | `combine_hidden_states`：`main_norm(main_proj(x))`（`dspark.py:201-202`） |

两边的 `+1` 是对称的：serving 请求 `id+1` 再取"`layer_idx+1` 命中时的层输出"，
净效果与我们"请求 id、取该层输出"相同。patch 里那句"Do not fix this to match V2"
是对的。

**特征契约不是接受率低的原因。**

### 4.2 24 个声明为 FP32 的参数在训练全程是 BF16 —— 中（已定性，非紧急）

模型显式声明为 FP32 的参数刚好 27 个，与官方 checkpoint 的 F32 计数一致：

| 参数 | 数量 | 声明处（`deepseek_v4_dspark.py`） | 实际 dtype |
| --- | ---: | --- | --- |
| `gate.bias`（router correction bias） | 3 | `:620-622` | **FP32**（`DeepseekV4Gate._apply:628-634` 在每次 `.to()` 后强制恢复） |
| `attn_sink` | 3 | `:370` | BF16 |
| `hc_attn_{fn,base,scale}` | 9 | `:889-894` | BF16 |
| `hc_ffn_{fn,base,scale}` | 9 | `:896-901` | BF16 |
| `hc_head_{fn,base,scale}` | 3 | `:928-933` | BF16 |

这解释了导出件实测的 `Counter({'BF16': 2375, 'F32': 3})`：**那 3 个 F32 就是三个
`mtp.{0,1,2}.ffn.gate.bias`**，不需要再跑诊断确认。

机理：`AutoDraftModel.from_config`（`specforge/modeling/auto.py:52-54`）在
`set_default_dtype` 构造之后还有一句 `model = model.to(dtype=torch_dtype)`，
把显式声明为 FP32 的**参数**一并降成 BF16。训练侧
（`algorithms/model_providers.py:205-208`）和导出侧
（`export/checkpoint_io.py:179`）走的是同一个函数，所以这不是导出时才掉的——
**训练全程就是 BF16**。

**这句 `.to()` 不能简单删掉**：训练用 FSDP1（`training/backend.py:317,349`），
其 flat parameter 要求单元内 dtype 一致，而这个模型故意不声明 `_no_split_modules`
（整模型一个 FSDP 单元），混合 dtype 的参数会直接报错。

严重性比初判低，两条理由：

1. **`gate.bias` 是 buffer，不进 flat param，代码已经单独保住了 FP32。**
   `after_optimizer_step` 里 `bias.add_(±0.001)` 的累加因此没有 BF16 的
   round-to-nothing 问题，负载均衡是正常工作的。
   （初版文档判断这里会失效，是错的。）
2. 剩下 24 个的**运算全程上采到 FP32**（`fn.float()` / `base.float()` /
   `scale.float()`，见 `:940-1000` 的 hc 前向），丢的是存储精度不是计算精度。

残留风险因此收敛为一个通用问题：**纯 BF16 训练、没有 FP32 master weights**，
AdamW 对这 24 个参数的小幅更新会有舍入损失。这不是 DSpark 特有的 bug，
要改得动 FSDP 的混合精度配置，不在本轮范围内。serving 侧无影响：
vLLM-Ascend 把它们声明为 FP32，加载时 `copy_` 自动上采。

### 4.3 没有接受率基线 —— 中

官方 drafter 在**同一台机器、同一组 prompt**下的接受率还没测。
没有它，第一步做完也无法判断成败。这件事和上机是同一次服务的两趟跑，成本很低。

### 4.4 第二步的量化 —— 低

见 §6。收益是显存/带宽，不是精度，晚做没有损失。

---

## 5. 首次上机结果（2026-09-02）与定位

### 5.1 数据

独立 draft 目录按 §2 起来了。跑了一批请求之后：

```
vllm:spec_decode_num_drafts_total                     439
vllm:spec_decode_num_accepted_tokens_per_pos{pos=0}    20
vllm:spec_decode_num_accepted_tokens_per_pos{pos=1..6}  0
```

| | 本 drafter | 官方 golden（e2e，w4a8/TP4，仅供量级参考） |
| --- | ---: | ---: |
| pos 0 | **4.6%**（20/439） | 0.88 |
| pos 1-6 | 0 | 0.74 / 0.58 / 0.49 / 0.40 / 0.30 / 0.18 |

### 5.2 这些数字说明什么

**pos 1-6 全 0 不是额外信息。** 每位置接受率约 4.5% 的话，pos1 的期望命中次数是
`439 × 0.045² ≈ 0.9`，观测 0 完全在噪声内。所以这不是"位置相关的 bug"，
而是**每个位置都均匀地接近无效**。

**4.6% 这个量级本身是有信息的。** 全随机 drafter 在 129280 词表下命中率约等于 0；
而一个退化输出（总是给高频 token：空格、换行、常见标点）大致就能蹭到百分之几。
所以现象是"**drafter 输出接近退化，不是稍弱**"——这把"训练不够、再多训点就好"
这个解释排掉了：从官方权重 warm start 的模型不会退化成这样，除非起点就不对
或者训练把它推走了。

### 5.3 已排除的原因

- **特征契约**（§4.1）：训练与 serving 两端的层号、mHC 折叠、投影逐行一致。
- **加载路径**：服务能起来，说明 draft 目录被解析、权重被加载、未被误判为量化模型。
- **dtype**（§4.2）：24 个参数是 BF16，但运算全程上采，且 serving 侧还会再上采一次。
  这个量级的偏差不可能把 88% 打到 4.6%。

### 5.4 嫌疑排序

**嫌疑 1：warm start 没有真正生效。** 追完 `_finish_registered_draft`
（`model_providers.py:158-190`）的三条分支，只有一条是静默的：

| 条件 | 行为 |
| --- | --- |
| 路径已配 且 不是 SpecForge checkpoint | `load_official_checkpoint` —— 缺任何权重都**抛错**，不会静默 |
| 路径已配 且 是 SpecForge checkpoint | `warm_start_draft_model` —— 同样会报 |
| **路径为空** | `_warm_start`（`:71-72`）**直接 return，什么都不打** → 随机初始化 |

也就是说，只要 `model.draft_checkpoint_path` 配到了，warm start 要么成功要么报错；
一旦为空，整个 run **没有任何证据**说明权重是随机的。100 步从零训出来的正是
"退化输出"这个量级。

> 已修（本次提交）：三条分支现在都打日志，第三条打的是 `WARNING ... RANDOM
> INITIALISATION`。以后翻一眼日志就能定这件事，不用再靠权重比对反推。

**嫌疑 2：学习率对一个已收敛的模型过大。** recipe 里是
`learning_rate: 2.0e-4`、`warmup_ratio: 0.04`、`max_steps: 10000`，
所以 warmup 是 400 步，step 100 时 lr ≈ `2e-4 × 100/400 = 5e-5`。
AdamW 每元素每步的位移量级约等于 lr，累计上界
`Σ lr_t ≈ 5e-7 × Σ_{t≤100} t ≈ 2.5e-3`；而 4096 宽线性层的权重标准差约
`1/√4096 ≈ 0.0156`。**上界相当于 16% 的相对位移**，而这期间只看过
`1 × 4 × 100 = 400` 条序列。这个 lr 是 EAGLE3 从零训的默认值，
不是给"微调一个已收敛的官方 drafter"用的。

（这是上界，梯度方向会互相抵消，所以它单独未必能把 88% 打到 4.6%；
但和嫌疑 1 叠加时它会放大后果。）

**嫌疑 3：100 步太少。** 只在嫌疑 1 成立时才是主因——warm start 生效的话，
100 步不该让接受率掉到 4.6%。

### 5.5 判据（按成本从低到高，做完 1-3 再上卡）

1. **训练日志里 step 1 的 `accuracy`（免费）。** warm start 生效的话，
   第一步的 accuracy 就该是高值；若从 ~0 开始往上爬，说明是从随机权重学起的，
   嫌疑 1 直接坐实。**注意 recipe 的 `log_interval: 20`**——第一条打印出现在
   step 20，不是 step 1；重跑时用 `training.log_interval=1`。
   顺带确认当时那个 run 的 `model.draft_checkpoint_path` 到底配的是什么。
2. **serving 日志里 `DSpark draft model loaded: N params`（免费）。**
   记下 N，确认没有大批权重没落位。
3. **离线比对导出件与官方权重**：

   ```bash
   python scripts/compare_draft_to_official.py \
       --draft-dir    /sharenfs/w00958190/dsv4-dspark/0901_export/deepseek-v4-flash-dspark-export \
       --official-dir /sharenfs/DeepSeek-V4-Flash-0731-w8a8
   ```

   它按 stage 抽 14 类张量 + 8 个 stage 专属张量，把官方侧的 W8A8 反量化后
   逐张量算 cosine 和相对 L2，最后给一个判定：

   | median cosine | 含义 |
   | --- | --- |
   | > 0.99 | warm start 生效，这是官方 drafter 的微调版 → 查嫌疑 2（学习率/目标函数） |
   | 0.05 – 0.5 | 已经和官方没什么关系 |
   | < 0.05 | **随机初始化**，warm start 从未生效 → 嫌疑 1 坐实 |

4. **官方基线**（占卡）：用 same-checkpoint 模式跑同一组 prompt，
   拿到这台机器上官方 drafter 的接受率。**这个数一直没测**，
   没有它就无法排除"这台机器上官方也不高"。

## 6. 第二步：量化

独立 draft 目录让第二步比原计划干净：**把 description 放进 draft 目录自己**，
不用碰 target。

```
deepseek-v4-flash-dspark-export-w8a8/
├── config.json
├── quant_model_weights.safetensors      # INT8 + weight_scale + weight_offset
├── quant_model_weights.safetensors.index.json
└── quant_model_description.json          # 2322×3 条 W8A8_DYNAMIC + 其余 FLOAT
```

`get_draft_quant_config` 会走 `VllmConfig.get_quantization_config(draft_model_config, ...)`，
draft 目录里有 description 就被识别为 ModelSlim（`quantization/utils.py:165`）。
启动时给 `--speculative-config` 加 `"quantization":"ascend"`。

量化规格（与 target 一致，实测确认）：

- 只量化那 774×3 = 2322 个矩阵（experts w1/w2/w3、shared_experts w1/w2/w3、
  attn wq_a/wq_b/wkv），其余标 `FLOAT`；
- per-output-channel 对称 INT8，`scale = absmax/127` 存 FP32 `[out,1]`，
  `offset` 全零，权重 clamp 到 **±127**（官方权重实测无 -128）；
- key 风格用**原生命名**（`mtp.0.attn.wq_a.weight`），与 target 的 description 一致。

自校验：导出后跑 `scripts/audit_deepseek_v4_dspark_checkpoint.py`；
再用 `load_official_checkpoint` 读回，与训练态 BF16 比相对误差。

---

## 7. 待办

**先做（不上机）**

1. §5.5 的判据 1-3：训练日志 step 1 的 accuracy、serving 日志的
   `DSpark draft model loaded` 计数、`compare_draft_to_official.py`。
   **这三条决定 4.6% 是加载问题还是训练问题。**
2. ~~4.1 的 mHC 折叠比对~~ / ~~4.2 的 dtype 诊断~~ —— 都已完成，见对应小节。
3. ~~导出侧写死 serving 用的 config 字段~~ —— 已完成：
   `scripts/prepare_dspark_serving_config.py`（含 8 个单测）。放在导出之后而不是
   `DeepseekV4DSparkConfig` 里，是因为训练侧要靠 `architectures` 解析模型类，
   `export_to_hf` 的输出也要能被 `from_pretrained` 读回。

**上机**

4. 官方 drafter（same-checkpoint）的接受率基线——仍然欠着，且现在更要紧了。
5. ~~跑 §2.3 的命令~~ —— 已跑，结果见 §5。

**之后**

6. §5 的量化导出器。

---

## 8. 本文没有验证的

1. ~~启动命令没实跑过~~ —— 已跑通（§5）。
2. 显存与吞吐未测；接受率只有这一个数据点，且**官方基线仍未测**——
   没有它就无法断定 4.6% 是"训练毁了它"还是"这台机器上官方也不高"。
3. ~~`captured.mean(dim=1)` 的正确性~~ —— 已排除（§4.1）。
4. §5 那三条判据一条都还没跑，所以"warm start 没生效"目前只是嫌疑最大的假设，
   不是结论。
5. 容器里实际安装的 `vllm_ascend` 是否逐字等于 `dedbb34`。用户确认用的是
   `fg11991/vllm-ascend@vllm-0.27-dsv4` 和 `fg11991/vllm@vllm-0.27-dsv4`，
   本文按此为准；但启动脚本里的 `VLLM_ASCEND_APPLY_DSV4_PATCH=1` 在这两个仓库里
   都搜不到读取处，该变量应为空转（DSpark patch 由
   `vllm_ascend/patch/worker/__init__.py:70` 无条件导入）。
