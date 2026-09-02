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
4. **剩下真正的风险不在 serving，在特征**：SGLang capture patch 里的
   `captured.mean(dim=1)` 这个 mHC 折叠**从未与官方实现对过**。它错了的话，
   drafter 学的是另一个目标，接受率不会好，而且和量化、和加载都无关。
5. 次要风险：模型声明为 FP32 的 27 个参数，在导出件里只剩 3 个是 F32。

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

`config.json` 必须改三个字段：

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

### 4.1 mHC 折叠 `mean(dim=1)` 从未对过官方实现 —— 最高

`patches/sglang/v0.5.14/apply_deepseek_v4_capture.py` 自己写着：

> NOT VERIFIED HERE: that `captured.mean(dim=1)` is the fold the official V4
> DSpark drafter consumes. It matches SpecForge's normalizer
> (`_project_target_hidden`, which does `mean(dim=-2)`), but the mHC stream mean
> is not the same operation as the learned `hc_head` fold.

采集侧把 `[T, hc_mult, H]` 的多流状态按 `mean` 折成 `[T, H]`，而官方 drafter 消费的
可能是 `hc_head` 学出来的加权折叠。**两者不同的话，drafter 训练时看到的 target 特征
和 serving 时 vLLM 喂给它的不是同一个东西**，接受率不会好，且这个失配在训练指标上
看不出来（loss 会正常下降，只是在拟合一个错的目标）。

排查方式：拿官方 `inference/model.py` 的 `hc_head` 前向，和 capture 出来的
`mean(dim=1)` 在同一批输入上比数值。这件事**应该在花力气做 serving 之前做**，
因为它决定这个 drafter 值不值得上机。

### 4.2 27 个 FP32 参数在导出件里只剩 3 个 —— 高

模型显式声明为 FP32 的参数刚好 27 个，与官方 checkpoint 的 F32 计数一致：

| 参数 | 数量 | 声明处（`deepseek_v4_dspark.py`） |
| --- | ---: | --- |
| `attn_sink` | 3 | `:370` |
| `gate.bias`（router correction bias） | 3 | `:621-625` |
| `hc_attn_{fn,base,scale}` | 9 | `:889-894` |
| `hc_ffn_{fn,base,scale}` | 9 | `:896-901` |
| `hc_head_{fn,base,scale}` | 3 | `:928-933` |

导出件实测 `Counter({'BF16': 2375, 'F32': 3})`。vLLM-Ascend 那边同样声明为 FP32
（`model.py:697-700`、`model.py:473`、`dspark.py:357-367`），加载时 `copy_` 会自动升回
FP32，不报错，但精度已经在导出时丢了。

最担心 `gate.bias`：`noaux_tc` 用它和 routed score 相加后选 top-6，
而 `router_bias_update_rate = 0.001` 的更新量在 BF16（8 位尾数）下很可能整个被舍掉
——**如果训练时它就是 BF16，那 router 的负载均衡在整个训练过程里可能都没生效**。

定位（两条，先跑第二条）：

```bash
# 导出件里哪 3 个还是 F32
python - <<'EOF'
import json, struct
with open('model.safetensors','rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n))
hdr.pop('__metadata__', None)
print('F32 :', [k for k,v in hdr.items() if v['dtype']=='F32'])
for pat in ('attn_sink','gate.bias','hc_attn_fn','hc_head_fn'):
    print(pat, ':', {v['dtype'] for k,v in hdr.items() if pat in k})
EOF

# 训练 checkpoint 里它们是什么 dtype —— 决定是训练掉的还是导出掉的
python - <<'EOF'
import torch
s = torch.load('<checkpoint>/training_state.pt', map_location='cpu', weights_only=False)
d = s['draft_state_dict']
for pat in ('attn_sink','gate.bias','hc_attn_fn','hc_head_fn'):
    print(pat, ':', {v.dtype for k,v in d.items() if pat in k})
EOF
```

- 训练 checkpoint 里是 FP32 → 只是导出时被 `export/checkpoint_io.py:179` 的
  `torch_dtype=torch.bfloat16` 一刀切了，改导出即可（保留声明为 FP32 的参数）。
- 训练 checkpoint 里就是 BF16 → 训练阶段掉的，要回去看 backend 的 dtype 处理，
  而且 4.1 之外又多一个"训练本身是否有效"的疑点。

### 4.3 没有接受率基线 —— 中

官方 drafter 在**同一台机器、同一组 prompt**下的接受率还没测。
没有它，第一步做完也无法判断成败。这件事和上机是同一次服务的两趟跑，成本很低。

### 4.4 第二步的量化 —— 低

见 §5。收益是显存/带宽，不是精度，晚做没有损失。

---

## 5. 第二步：量化

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

## 6. 待办

**先做（不上机）**

1. 4.1 的 mHC 折叠数值比对。**这是决定 drafter 值不值得上机的那一条。**
2. 4.2 的两条 dtype 诊断。
3. 导出侧写死 `architectures: ["DSparkDraftModel"]` 和 `n_mtp_layers: 3`
   （改在 `DeepseekV4DSparkConfig`，不要手改 json）。

**上机（一次服务跑两趟）**

4. 先跑官方 drafter（same-checkpoint），记下接受率——这是基线。
5. 再跑 §2.3 的命令，比同一组 prompt 的接受率。

**之后**

6. §5 的量化导出器。

---

## 7. 本文没有验证的

1. §2 那条独立 draft 目录的启动命令**没有实跑过**。源码链条是通的，但
   `ModelConfig` 构造阶段、KV cache 分配、`dspark_head` 的 compile tag 这些
   都可能有没读到的分支。
2. 接受率、显存、吞吐一律未测。
3. `captured.mean(dim=1)` 的正确性（这正是 4.1）。
4. 容器里实际安装的 `vllm_ascend` 是否逐字等于 `dedbb34`。用户确认用的是
   `fg11991/vllm-ascend@vllm-0.27-dsv4` 和 `fg11991/vllm@vllm-0.27-dsv4`，
   本文按此为准；但启动脚本里的 `VLLM_ASCEND_APPLY_DSV4_PATCH=1` 在这两个仓库里
   都搜不到读取处，该变量应为空转（DSpark patch 由
   `vllm_ascend/patch/worker/__init__.py:70` 无条件导入）。
