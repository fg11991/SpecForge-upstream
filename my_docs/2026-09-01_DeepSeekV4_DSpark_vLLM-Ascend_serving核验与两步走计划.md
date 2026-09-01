# DeepSeek-V4 DSpark 微调权重回灌 vLLM-Ascend：源码核验与两步走计划

- 日期：2026-09-01
- 分支：`dsv4-moe-dspark-trainable`
- 文档编写时分支 tip：`6d2b5e5`
- 核验对象：`fg11991/vllm-ascend`，分支 `vllm-0.27-dsv4`，commit `dedbb34`
  （"vLLM Ascend 0.27 DeepSeek-V4 snapshot"，浅克隆只读）
- 方法：纯静态。只读两边源码。**没有下载任何权重**，也没有上机。
  HF 在本环境不可达（代理 403），凡是需要读 checkpoint 才能确认的事项，
  本文只给判定命令，不给结论。

---

## 0. 结论先行

1. 外部那份"vLLM-Ascend 加载逻辑与 0731 W8A8 格式"的分析，**逐条对得上源码**，
   数字也和我们 `compatibility_report_dsv4_dspark.md` 的独立统计精确闭合。
2. 但它默认了一件没验证的事：`quant_model_description.json` 的 key 就是磁盘张量名。
   vLLM-Ascend 的查表路径指向另一种命名。**如果它错了，我们的 ModelSlim warm start
   会静默加载错权重，审计脚本也不会报**。判定命令见 §4。
3. 微调后的 drafter 回灌有**两条**路，不是只有"重新量化"一条：把 `mtp.*` 标成
   `FLOAT` 就能以 BF16 直接 serving，这是 vllm-ascend 对 `deepseek_mtp` 的官方做法。
4. 因此"先 BF16 上机、再量化"的两步走成立，而且推荐：它把"训练有没有效"和
   "量化掉不掉点"分成两个独立可判定的问题。
5. 当前导出件有两个硬问题挡在第一步前面：单个 safetensors（疑似 expert 丢失，§5）
   和 `architectures` 名字 vLLM 不认（§6）。

---

## 1. vLLM-Ascend DSpark 加载路径：逐条核实

以下行号均指 `fg11991/vllm-ascend@dedbb34`。

### 1.1 same-checkpoint 名字映射（成立）

`vllm_ascend/models/deepseek_v4/dspark.py:503-527` 的 `_remap_dspark_name`：

```
mtp.(\d+).(rest)  ->  model.layers.{num_hidden_layers + stage}.{rest}
```

外加四类特例：

| 磁盘名 | 运行时名 |
| --- | --- |
| `mtp.*.main_proj` / `main_norm` | 固定归 `layers.43` |
| `mtp.*.norm` / `markov_head.*` | 固定归 `layers.45` |
| `mtp.*.confidence_head.*`、`hc_head_{fn,base,scale}` | 提到 `model.*` |
| `mtp.0.embed.weight` / `mtp.2.head.weight` | `model.embed_tokens.weight` / `lm_head.weight` |

随后做子串替换 `.attn.→.self_attn.`、`.ffn.→.mlp.`、`.w1/.w2/.w3→.gate_proj/.down_proj/.up_proj`、
`.mlp.gate.bias→.mlp.gate.e_score_correction_bias`。

**对我们的意义：SpecForge 导出的原始命名（`mtp.0.attn.wq_a.weight`、
`mtp.0.ffn.experts.7.w1.weight`）正是它要吃的形式，不需要在导出侧改名。**

### 1.2 关键机制：draft 层用的是 target 的 hf_config

`dspark.py:139-146` 构造三个 stage：

```python
str(self.mtp_start_layer_idx + idx): DeepseekV2DecoderLayer(
    vllm_config, prefix=f"mtp.{idx}", is_draft_layer=True,
)
```

`config` 参数缺省，`models/deepseek_v4/model.py:656-657` 回落到
`vllm_config.model_config.hf_config` —— **target 的配置**。只有
`DeepseekV4DSparkModel.__init__`（`dspark.py:122`）读
`speculative_config.draft_model_config.hf_config`。

推论（对导出侧很重要）：

- `compress_ratios`、`num_hash_layers`、`index_topk_*`、`compress_rope_theta`
  这些 draft config 里没有的字段**不需要补**，它们从 target 来；
- draft config 只需提供 DSpark 级字段，清单见 §6。

另注意 `dspark.py:139` 的 `prefix=f"mtp.{idx}"` 与 ModuleDict 的 key `str(43+idx)`
是**两套命名**：参数树是 `model.layers.43.*`，而量化查表用的 prefix 是 `mtp.0.*`。
这是故意的，§4 依赖这一点。

### 1.3 RoPE 不会走错（好消息）

`vllm_ascend/utils.py:87-100` `extract_dsv4_layer_index` 把 `mtp.0` 映射成
`num_hidden_layers + 0 = 43`；`utils.py:110-115` `get_dsv4_compress_ratio` 对
超出 `compress_ratios` 长度的层返回 0（注释原文："treating unspecified MTP
layers as dense"）。`model.py:519-521,525` 因此走 `compress_ratio == 0` 分支：
SWA、`rope_parameters["rope_theta"] = config.rope_theta`，不启 yarn。

**与我们 `DeepseekV4DSparkAttention` 硬编码 `rope_scaling = None`、theta 10000
的训练假设一致。** 导出 config 里那两坨 yarn 字段是死字段，不会被 draft 层读到。

### 1.4 draft 继承 target 量化配置的补丁（成立）

`vllm_ascend/patch/worker/patch_v2/patch_dspark.py`：判据是

```python
inherits_target_quant = draft_model_config.model == vllm_config.model_config.model
```

命中就把 `model_utils.get_draft_quant_config` 临时换成返回 `vllm_config.quant_config`，
`finally` 里还原。由 `patch/worker/__init__.py:70` **无条件 import**。
不打补丁的失败模式是 KeyError（draft linear 缺 `weight_scale`/`weight_offset`/
`scale_bias`），不是静默。

**反向推论**：draft 路径与 target 路径不同时，这个补丁不生效，draft 按未量化建层
—— 这正是 §7 路 1a 想要的。

### 1.5 `VLLM_ASCEND_APPLY_DSV4_PATCH` 是空转的

全仓 grep 该变量 **0 命中**，`vllm_ascend/envs.py` 里也没有。这个分支无条件导入
DSpark patch。启动脚本里那行 export 不起任何作用（旧镜像遗留或私有补丁开关）。

> 但这只证明**这份源码**不读它。容器里装的是不是这份代码要另行确认：
> `python -c "import vllm_ascend, os; print(os.path.dirname(vllm_ascend.__file__))"`
> 然后比对 `patch/worker/patch_v2/patch_dspark.py` 是否存在及内容。

### 1.6 官方用法全是 same-checkpoint

`docs/source/tutorials/models/DeepSeek-V4-Flash.md` 与
`tests/e2e/pull_request/four_card/spec_decode/test_dspark_deepseekv4.py` 里的
speculative config 一律是

```json
{"method": "dspark", "num_speculative_tokens": 7, "enforce_eager": true}
```

不带独立 draft 路径；e2e 用的权重是 `DeepSeek-V4-Flash-DSpark-w4a8-test`。
**独立 draft 目录属于没有先例的走法**，§7 据此排优先级。

---

## 2. ModelSlim W8A8 数量核对（闭合）

用官方 FP4/FP8 checkpoint 的统计（`compatibility_report_dsv4_dspark.md` §1）反推
W8A8 版本的条目数：

| 量 | 推导 | 外部说法 |
| --- | --- | ---: |
| 每 stage 待量化矩阵 | 256×3 experts + shared_experts 3 + `wq_a/wq_b/wkv` 3 = 774 | — |
| 三 stage 合计 | 2322 | — |
| `W8A8_DYNAMIC` 条目 | 2322 ×（weight + weight_scale + weight_offset）= **6966** | 6966 |
| `FLOAT` 条目 | 官方 2376 权重 − 2322 = 54，加 `mtp.0.embed`/`mtp.2.head` = **56** | 56 |
| `mtp.*` 合计 | **7022** | 7022 |

官方侧同样闭合：FP4 2304 + FP8 25 = 2329 个 scale，与 4705 = 2376 + 2329 一致。

量化形式（对称 per-output-channel INT8、`scale` 形状 `[out,1]`、offset 恒 0）
与本仓库 `deepseek_v4_dspark.py:1102-1130` 的实现一致，三条独立证据见
`my_docs/2026-08-18_..._A3穿刺与A2上机手册.md` §2.9。"权重行 min=-127、无 -128"
对加载方向无影响，但对**重新量化导出**是硬约束（必须 clamp 到 ±127）。

---

## 3. 本仓库当前能力边界

- **已有**：官方 FP4/FP8 与 ModelSlim W8A8 两种 warm start
  （`deepseek_v4_dspark.py::load_official_checkpoint`）、权重审计脚本、
  Ascend EP 训练、HF/SGLang draft 导出。
- **没有**：把训练完的 `mtp.*` 写回 fused checkpoint、重新量化、写 description。
  开发文档 §6.3 与 §10 第 5 条已声明这是"下一阶段 serving adapter"。

本文讨论的正是这块空白。

---

## 4. 未决项（最高优先级）：description 的 key 命名

我们的 loader（`deepseek_v4_dspark.py:1569`）这样查标签：

```python
label = quant_description.get(checkpoint_name)   # checkpoint_name = 磁盘张量名
```

**隐含假设：description 的 key 与 safetensors 张量名逐字相同。**

而 vLLM-Ascend 侧：模块 prefix 是 `mtp.0.self_attn.wq_a` /
`mtp.0.mlp.experts.0.gate_proj`（`model.py:474-516,672-686`），
`quantization/configs/modelslim_config.py:736-742` 又把
`QUANT_MODEL_SUBSTR_MAPPINGS["deepseek_v4"]`（`.attn.→.self_attn.`、
`.w1.→.gate_proj.`、`.ffn.→.mlp.`）**作用在 prefix 上**（方向是原始名→vLLM 名），
而 `models/deepseek_v4/` 下没有定义 `hf_to_vllm_mapper`，description 的 key
不会被规范化。两边要对上，description 的 key 就得是 `self_attn/mlp/gate_proj` 风格。

后果不对称：

- **vLLM-Ascend 侧会响**：非 fused 层 `create_scheme_for_layer` 查不到；
  fused 层 `is_layer_skipped_ascend`（`modelslim_config.py:854`）直接 KeyError。
- **我们训练侧是哑的**：`quant_description.get()` 全返回 `None` → 走 FLOAT 分支 →
  找不到 `weight_scale_inv` → **把 INT8 原样 copy 进 BF16 参数，shape 检查还能过，
  不报任何错**。
- **审计脚本也是哑的**：`scripts/audit_deepseek_v4_dspark_checkpoint.py:340-345`
  是拿 description 的 key 去过滤实际张量的，对不上就返回空列表，
  而空列表在退出码判定（`:528-531`）里不算失败。

### 判定命令（在有 checkpoint 的机器上跑）

```bash
python - <<'EOF'
import json
root = '/sharenfs/DeepSeek-V4-Flash-0731-w8a8'
d = json.load(open(f'{root}/quant_model_description.json'))
ks = [k for k in d if k.startswith('mtp.')]
print('mtp entries      :', len(ks), '(期望 7022)')
print('sample           :', ks[:6])
print('self_attn style  :', sum('.self_attn.' in k for k in ks))
print('attn style       :', sum('.attn.' in k and '.self_attn.' not in k for k in ks))
idx = json.load(open(f'{root}/quant_model_weights.safetensors.index.json'))['weight_map']
print('key == tensor 名 :', len(set(ks) & {k for k in idx if k.startswith('mtp.')}), '(期望 7022)')
EOF
```

最后一行是判据：**等于 7022 → 现有代码是对的；等于 0 → 那条 ModelSlim 路径从未
真正生效，所有以它 warm start 的训练结果都要作废重跑。**

不论结果如何，都应补两处防护：

1. loader：description 存在但 `mtp.*` 一个 key 都没命中时报错，并打印实际 key 样例；
   同时加一层 key 规范化（`attn↔self_attn`、`ffn↔mlp`、`w1/w2/w3↔gate_proj/up_proj/down_proj`
   双向尝试），命中哪种打一行日志。
2. 审计脚本：`checkpoint_format == "modelslim"` 且 `quantized_samples` 为空 → 判失败。

---

## 5. 导出件核验：单个 safetensors 是可疑的

BF16 三阶段 DSpark ≈ **19.85 B 参数 ≈ 39.7 GB**，加上模型自带的
`mtp.0.embed` + `mtp.2.head`（各 129280×4096 BF16 ≈ 1.06 GB）≈ **41.8 GB**。
`save_pretrained` 默认 `max_shard_size="5GB"`，正常应产出约 9 个分片 + index.json。

只有一个文件，除非显式改过 `max_shard_size`，否则最可能是
**EP 的 expert 权重没被合并进来**：`training/controller.py:899-903` 把
`.ffn.experts.` 的张量单独写进各 rank 的 rank file，只有
`export/checkpoint_io.py:150-160` 的 `_consolidate_export_state` 在
`expert_parallel_size > 1` 时才会拼回来。

```bash
python - <<'EOF'
import json, struct, collections, glob
p = sorted(glob.glob('*.safetensors'))[0]
with open(p,'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))
hdr.pop('__metadata__', None)
size = sum(v['data_offsets'][1]-v['data_offsets'][0] for v in hdr.values())
print('file       :', p)
print('tensors    :', len(hdr), '(期望 2378)')
print('experts    :', sum(1 for k in hdr if '.ffn.experts.' in k), '(期望 2304)')
print('bytes      : %.1f GiB (期望 ~41.8)' % (size/2**30))
print('dtypes     :', collections.Counter(v['dtype'] for v in hdr.values()))
print('embed/head :', [k for k in hdr if k.endswith(('embed.weight','head.weight'))])
EOF
```

`experts` 不是 2304 就先修导出，别往下走。

---

## 6. 导出 config.json 字段对照

按 §1.2，draft config 只需提供 DSpark 级字段：

| vLLM-Ascend 读取处 | 字段 | 当前导出 | 判定 |
| --- | --- | --- | --- |
| `dspark.py:129` | `num_hidden_layers` → `mtp_start_layer_idx` | 43 | ✅ 必须保持 43，不能写成 3 |
| `dspark.py:123-127,176-178` | `hc_mult` / `hc_eps` / `rms_norm_eps` / `hidden_size` / `vocab_size` | 有 | ✅ |
| `dspark.py:126-127` | `dspark_block_size` / `dspark_target_layer_ids` | 5 / [40,41,42] | ✅ |
| `patch_speculative_config.py:46-47` | `dspark_noise_token_id` → `ptd_token_id` | 128799 | ✅ 自动补 |
| `dspark.py:65-69` | `n_mtp_layers`，其次 `dspark_num_mtp_layers`，再默认 3 | 只有 `dspark_num_layers` | ⚠️ 靠默认值蒙对，应显式写 `n_mtp_layers: 3` |
| `models/__init__.py:39-42` | `architectures` | `DeepseekV4DSparkDraftModel` | ❌ **注册名是 `DSparkDraftModel`**，现名 vLLM 找不到模型 |
| `modelslim_config.py:383,724` | `model_type`（决定 packed/substr 映射） | `deepseek_v4_dspark` | ⚠️ 表里只有 `deepseek_v4`；走 FLOAT 路无影响，走 W8A8 时要改或补映射 |

`architectures` 与 `n_mtp_layers` 建议改在 `DeepseekV4DSparkConfig` 的导出侧，
而不是手工改 json。

> 命名先例：`patch_speculative_config.py:28` 对 qwen3 dspark 的判据也是
> `"DSparkDraftModel" in architectures`，说明这是该分支统一的 draft 架构名。

---

## 7. 两步走方案

### 为什么分两步

- 第一步（BF16）验的是**训练本身有没有效**：接受率相对官方 drafter 涨了没有。
- 第二步（量化）验的是**量化掉了多少点**。
- 合在一起做，接受率不及预期时无法区分是训练没学到东西还是量化砸了。
- 量化的收益是显存/带宽而非精度，晚做没有损失。

**注意精度名称**：导出是 **BF16**（`checkpoint_io.py:179` `torch_dtype=torch.bfloat16`，
config `dtype: bfloat16`），不是 FP16，且必须保持 BF16 —— 启动配方里有
`INF_NAN_MODE_FORCE_DISABLE=1`，FP16 溢出不会报错，只会悄悄出 NaN。

### 第一步的两条子路

**路 1a：独立 draft 目录（首选）**

```
--speculative-config '{"method":"dspark","model":"/path/to/bf16-draft",
                       "num_speculative_tokens":7,"enforce_eager":true}'
```

好处：draft 路径 ≠ target 路径 → §1.4 的补丁不生效 → draft 按未量化建层 →
BF16 直接加载，description 一个字都不用动。且我们的模型自带
`mtp.0.embed` / `mtp.2.head`（`deepseek_v4_dspark.py:902-925`，`642c237` 引入），
经 §1.1 的映射正好落到 draft 自己的 `embed_tokens` / `lm_head`
（`dspark.py:131,323`），**目录是自洽的，不需要额外注入 embedding**。

能否走通取决于上游一个函数，容器里一条命令可判：

```bash
python -c "import vllm.v1.worker.gpu.spec_decode.dspark.utils as u, inspect; \
print(inspect.getsource(u.load_dspark_model))"
```

看它是从 `draft_model_config` 取 loader，还是硬用 target 的权重迭代器。
前者 → 1a 可行；后者 → 走 1b。考虑到 §1.6（官方用法全是 same-checkpoint），
1a 失败是正常结果，不要在上面耗太久。

**路 1b：覆盖回 target 目录（一定能通）**

把 BF16 的 `mtp.*` 写成新分片放进 target 目录、更新 index，并把
`quant_model_description.json` 里所有 `mtp.*` 条目标成 `FLOAT`。
`is_layer_skipped_ascend`（`modelslim_config.py:834-871`）见到 FLOAT 就返回
`AscendUnquantizedLinearMethod` / `AscendUnquantizedFusedMoEMethod`，BF16 正常加载。

这正是 vllm-ascend 自己对 `deepseek_mtp` 的官方做法，见
`modelslim_config.py:176-178` 的注释原文：

> NOTE 1. The quantized MTP layer of deepseek on the NPU is not quantized;
> NOTE 2. The description file generated by the current msmodelslim tool does not
> have MTP layer info. Please manually add it and set the value to FLOAT.

对照：SGLang 侧配方里的 `FORCE_DRAFT_MODEL_NON_QUANT=1` 是同一件事。

实施注意：用软链接建一个新目录，只把改动的分片和 description 落到新目录，
不要污染原始 target 权重；备份 `ori_quant_model_description.json`
（`examples/save_sharded_state_310.py:252` 就是这个约定）。

### 显存

BF16 drafter 在 EP8 下约 **5 GB/rank**，比 W8A8 多约 2.5 GB/rank。
实测 target 权重 36.99 GB/rank、卡上尚空 23.8 GB，放得下。

### 第二步（量化）

`specforge/export/to_vllm_ascend_dspark.py`，`--draft-precision {float,w8a8}`：

- 合并 EP rank-local experts → 完整 `mtp.*`（复用 `materialize_draft`）；
- `float`：写 BF16 `mtp.*`，description 中 `mtp.*` 全标 FLOAT；
- `w8a8`：只量化 §2 那 774×3 个矩阵，其余保持 FLOAT；
  `scale = absmax/127` FP32 `[out,1]`、`offset` 全零、权重 clamp ±127 存 INT8；
- **key 风格跟随现有文件**：读入原 description，学它的命名风格再写 `mtp.*` 条目，
  这样 §4 的结论无论是哪个都不会写错；
- 自校验：导出后跑审计脚本；再用 `load_official_checkpoint` 读回，
  与训练态 BF16 比相对误差（float 路应为 0，w8a8 路期望 1e-2 量级）。

---

## 8. 两个待堵的静默口子

1. **`materialize_draft` 对任何含 "embed" 的缺失键一律容忍**
   （`export/checkpoint_io.py:187-191`），而 `export_to_hf` 那条"缺 embedding 就报错"
   的守卫判的是 `hasattr(model, "embed_tokens")`（`to_hf.py:91`）——
   我们的名字是 `mtp.0.embed`，**这个守卫对 DSpark 根本不触发**。
   目前因为 `DSparkTrainStrategy.checkpoint_state_filter` 保留了全部
   `draft_model.*`（含冻结的 embed/head），实际没出事，但这是个一改就中的雷。
2. §4 的 description key 防护（loader + 审计脚本各一处）。

另：`my_docs/2026-08-18_..._A3穿刺与A2上机手册.md` §2.9 中"审计报告的
`unexpected: [mtp.0.embed.weight, mtp.2.head.weight]` 是预期的"一句已过时 ——
`642c237` 之后模型自己声明了这两个张量，`unexpected` 应为空。

---

## 9. 行动清单

**Step 0（不写代码，先做）**

1. 跑 §5 的 header 检查，确认导出件完整（`experts == 2304`）。
2. 跑 §7 的 `inspect.getsource(load_dspark_model)`，定 1a / 1b。
3. 跑 §4 的判定命令，定 description key 命名。
   —— 只影响第二步和 warm start 正确性，**不挡第一步**。
4. 确认容器里的 `vllm_ascend` 是否就是 `dedbb34`（§1.5）。
5. 确认最终部署形态是 W8A8 还是 W4A8_MXFP。我们的 loader 遇到 MXFP 标签是报错
   而非猜（`deepseek_v4_dspark.py:1570-1576`），若目标是 W4A8 则读写两侧都要新增支持。

**Step 1（BF16 上机）**

1. 修 `architectures` → `["DSparkDraftModel"]`，加 `n_mtp_layers: 3`（改在导出侧）。
2. 按 1a 或 1b 起服务。
3. 判据：`vllm:spec_decode_num_accepted_tokens_per_pos` 与官方 drafter 的 golden
   `[0.88, 0.74, 0.58, 0.49, 0.40, 0.30, 0.18]`（e2e 测试里的数）对比。
   不低于 → 训练链路成立；明显低 → 是训练/特征问题，不要去怪量化。

**Step 2（量化）**：Step 1 通过后按 §7 末节实施。

**贯穿**：Step 1 之前把 §8 两个静默口子堵上。

---

## 10. 本文没有验证的东西

诚实边界，避免后来者误读：

1. `quant_model_description.json` 的实际 key 命名（HF 不可达，§4 是判定方法不是结论）。
2. 上游 `load_dspark_model` 是否支持独立 draft 路径（本环境未安装 vllm）。
3. 容器内实际安装的 `vllm_ascend` 是否等于 `dedbb34`。
4. 任何真机数值：接受率、显存、吞吐一律未测。
5. 上机总闸门 `custom_ops` 缺件（见 2026-08-18 手册 §1）仍未解决，与本文这条线
   相互独立，但它不通则 Step 1 无从谈起。
