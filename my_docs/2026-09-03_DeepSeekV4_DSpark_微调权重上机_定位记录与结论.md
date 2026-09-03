# DeepSeek-V4 DSpark 微调权重上机：定位记录与结论

- 日期：2026-09-02 ~ 2026-09-03
- 分支：`dsv4-moe-dspark-trainable`
- 核验对象：`fg11991/vllm-ascend@dedbb34`、`fg11991/vllm@3d04f15`（本地已切到这两个 commit 逐行核对）
- 与 [2026-09-01 那份](2026-09-01_DeepSeekV4_DSpark_vLLM-Ascend_serving核验与两步走计划.md) 的关系：那份是**上机计划**，这份是**结果**。两份的结论冲突时以本文为准。

---

## 0. 结论先行

1. **路线定了：独立 draft 目录 + W8A8**。官方权重原样放进独立目录、以 INT8 服务，与 same-checkpoint **实测等效**（gsm8k-200：55.8% vs 55.4%）。
2. **BF16 的独立 draft 目录不可用**。同一批官方权重反量化成 BF16 放进同一个目录，接受率 **< 1%**。这是整件事唯一的病灶，前两天所有"微调把 drafter 毁了"的观测都是经由这条坏路径测出来的。
3. **0901 §3 的 same-checkpoint 覆盖方案下架**，不用重写混合分片、不用碰那份 294 GiB 的 checkpoint。
4. **基线**（gsm8k-200、greedy、`num_speculative_tokens=7`、concurrency 4）：**SpecAcc 55.4%、AvgSpecLen 4.88、TPOT 22.64 ms**。微调版要打的是这个数。
5. **微调后的 drafter 至今没有被公平评测过。** 4.6% 那个数不作数。
6. 量化导出器已落地：`scripts/export_dspark_ascend_w8a8.py`。

---

## 1. 时间线：做了什么，得到了什么

| # | 实验 | 目的 | 结果 | 结论 |
| --- | --- | --- | --- | --- |
| 1 | 导出件 vs 训练态的 dtype 审计 | 查导出是否掉精度 | 一致 | 导出没损坏 dtype；但 24 个声明为 FP32 的参数**训练全程**是 BF16 |
| 2 | `compare_draft_to_official.py` | 查 warm start 是否生效 | median cosine 0.99988 | **后被证伪**，见 §2.1 |
| 3 | 官方 same-checkpoint 基线 | 建立对照 | 单请求日志 58.3% / 65.4% | 口径不可靠，见 §2.4；后由 200 条基准修正为 55.4% |
| 4 | draft config 有效值逐条核对 | 查 0901 §5.5(a) | 四条解析链两边**落点全同** | (a) 排除 |
| 5 | `DSpark draft model loaded: N params` | 查 0901 §5.5(c) | 官方 119 / 我们 77，差 42 | 42 = W8A8 独有的 scale/offset 参数，**两边都是完整加载**，(c) 排除 |
| 6 | `check_dspark_serving_names.py` | 查名字映射 | 2378/2378 映射，零丢弃零撞名 | 加载路径关闭 |
| 7 | 三个 `gate.bias` 覆回官方 | 查 0901 §5.5(d) | 接受率**无变化** | router 漂移不是主因 |
| 8 | 32 个小张量（`attn_sink`/`hc_*`/各 norm）覆回官方 | 查 Adam 位移 | 接受率**无变化** | 小张量位移不是主因 |
| 9 | **identity export**：官方权重反量化成 BF16 | 把"训练毁的"和"serving 毁的"分开 | **< 1%** | **(b) 坐实**：BF16 未量化路径不可用；微调侧洗清 |
| 10 | **int8_draft**：官方 INT8 分片软链进独立目录 | 独立目录本身行不行 | 200 条基准 **55.8%** | 独立目录 + INT8 **可用且无损耗** |
| 11 | 反量化 → 再量化往返自检 | 验证量化约定 | `max\|dq\|=1`、`rel_ds = 1/127` | 约定确认：`absmax/127`、对称、per-out-channel |
| 12 | 读 msmodelslim PR #757 | 查官方量化配方 | QuaRot 只保留 `mtp.0.main_proj` 右旋；embed/head 用**未旋转**副本 | 见 §3.5 |

### 1.1 最终对照（200 条 gsm8k，greedy，同一命令）

| | SpecAcc | AvgSpecLen | pos0..pos6 | TPOT |
| --- | ---: | ---: | --- | ---: |
| 原生 same-checkpoint | 55.4% | 4.88 | 89.7 / 77.7 / 66.6 / 54.4 / 44.0 / 32.8 / 22.6 | 22.64 ms |
| 官方权重 + 独立目录 + INT8 | **55.8%** | **4.91** | 90.6 / 78.6 / 66.8 / 55.2 / 44.3 / 32.4 / 22.6 | 23.00 ms |
| 官方权重 + 独立目录 + BF16 | **< 1%** | — | — | — |

前两行差 0.4 个点，在噪声内。第三行是同一批权重、同一个目录、同一条命令，只换成未量化 kernel。

---

## 2. 被证伪的五个结论

这一节比结论本身有用：它们都曾看起来很有道理。

### 2.1 "median cosine 0.99988，所以权重没问题"

`compare_draft_to_official.py` 的 cosine 列**数值失真**。两个与官方**逐位相同**的张量（`mtp.0.embed.weight`、`mtp.2.head.weight`，`relative_l2` 精确为 0）报出 1.1846 和 1.2202——余弦相似度不可能大于 1。

机理：`cosine_similarity` 在 fp32 里累加 `sum(x*x)`，5.3 亿个 ~4e-4 量级的项累加到 ~2e5 后，fp32 该量级的 ulp 是 0.0156，后续项被整个吞掉，两个范数被系统性低估。本地复现（`cos(a, a)`，理论恒为 1）：

| 元素数 | 1e6 | 1e7 | 1e8 |
| --- | ---: | ---: | ---: |
| fp32 `cosine_similarity` | 1.000069 | 1.001731 | 1.060107 |

已修（`c96c113`）：全部归约改 float64 分块累加，verdict 改用 `relative_l2`，抽样补上 `ffn.gate.bias` 和每个 EP rank 各一个 expert，报告直接给 `worst_by_relative_l2`。

### 2.2 "router 的 `gate.bias` 漂移是主因"

实测漂移确实很大：`max|Δ| = 0.107/0.107/0.115`（正好是 100 步 × 0.001 的上界，`sign()` 一次没翻号），rms 是官方分布宽度的 **34%~72%**，平均把每个 expert 在路由排序里挪 **70~77 个名次**（共 256）。

推理链看着无懈可击——直到把三个 bias 覆回官方值重起服务：**接受率没有任何变化**。

### 2.3 "小张量的 Adam 位移是主因"

也确实存在：微调对**每个**张量的绝对位移都在 0.004~0.011，与该张量自身量级无关（官方 absmax 从 0.059 到 17.1 跨 290 倍，位移只跨 2 倍）——这是 Adam 的指纹（每步位移≈lr，与参数大小无关；100 步 × 2e-4 带 warmup/衰减 ≈ 0.005~0.01）。对小量级参数就是重写：`mtp.2.hc_head_fn` 相对 L2 动了 **29.4%**。

把 32 个小张量全部覆回官方值：**接受率仍然没有变化**。

### 2.4 "独立目录要付 10 个点"

来自单请求的滑动窗口日志（`metrics.py` 每 10 秒一行）：官方 58.3% / 65.4%，独立目录 49.9% / 48.5%。**官方那两个窗口自己就差了 7 个点**——样本是一个请求。换成 200 条基准后差距归零。

**方法论结论：所有接受率对比一律走 200 条基准脚本，禁止用日志里的滑动窗口值下判断。**

### 2.5 "`has_own_embed_tokens = quant_config is not None` 会让 drafter 用 target 的 embed/head"

这是 `v0.26.0rc_dspark_dev` 分支的实现，**dedbb34 上不是这样**：`DSparkDeepseekV4ForCausalLM` 不设这两个 flag，而是在 `load_weights` 里调 `process_eagle_weight(self, name)`（`dspark.py:450`），看到 remap 后的 `model.embed_tokens.weight` / `lm_head.weight` 才置位；`_should_share` 于是走到 `torch.equal` 比较，两者不等 → drafter 保留自己的。0901 §2.4 在这一版上是对的。

**教训：核对源码前先确认本地 checkout 与容器同 commit。**

---

## 3. 已确认的事实（源码级）

### 3.1 独立目录模式下，draft config 驱动什么

`DeepseekV4DSparkModel.__init__` 第一行是 `config = vllm_config.speculative_config.draft_model_config.hf_config`（`models/deepseek_v4/dspark.py:118-129`），所以外层 drafter 的每个尺寸和开关都来自 draft 目录的 config；三个 decoder layer 是 `config=None` → 回落到 **target 的 hf_config**。

四条**不改变张量形状、因而不报错**的解析链，全部在 vLLM 侧（`diff_dspark_serving_config.py` 覆盖不到）：

| 有效值 | 位置 | 我们的 draft 命中 | target 命中 |
| --- | --- | --- | --- |
| ptd / mask token | `gpu/spec_decode/utils.py:55-77` | `dflash_config.mask_token_id` = 128799 | `dspark_noise_token_id` = 128799 |
| aux 层 | `gpu_model_runner.py:5503-5536`（**先于** `model_runner_v1.py:675-689`） | `dflash_config.target_layer_ids`+1 → (41,42,43) | `dspark_target_layer_ids`+1 → (41,42,43) |
| `sample_from_anchor` | `gpu/spec_decode/dspark/speculator.py:46-52` | 缺 → 默认 True | 缺 → 默认 True |
| `use_non_causal` | `dspark/utils.py:28` + `models/qwen3_dflash.py:58-72` | 无 `layer_types`/`causal` → True | 同左 |

补充：`get_dsv4_compress_ratio`（`utils.py:110-115`）在字段缺失时返回 0，而 target 的 `compress_ratios[43..45]` 正好是 0,0,0；`dspark_block_size` 在 dedbb34 里只被存进 `self.block_size` 后从未读取。

**为什么 draft config 缺 `num_hash_layers` 不炸**：`initialize_model`（`vllm/model_executor/model_loader/utils.py:41-62`）只用传入的 `model_config` **选模型类**，构造时传的是 `draft_vllm_config`，而 `load_dspark_model` 的 `replace(...)` 没有改 `model_config`——decoder layer 里的 `vllm_config.model_config.hf_config` 仍是 target 的。

### 3.2 加载完整性：119 = 77 + 42

`loaded_params` 数的是去重后的参数名。

- **77**（BF16 独立目录）：2378 个张量 = 2304 expert + 74 非 expert；expert 塌缩成每 stage 2 个名字（6），非 expert 里每 stage 的 `w1`/`w3` 合并进 `shared_experts.gate_up_proj.weight`（−3）→ 71 + 6 = 77。
- **119**（same-checkpoint）：77 + 每 stage 14 个 scale/offset × 3 = 42。14 = attn 三个矩阵各 2 + `shared_experts` 两个参数各 2 + experts 两个参数各 2，与 `quantization/methods/w8a8/w8a8_dynamic.py:81-82,210-217` 注册的参数一一对应。

加载分支全是 `params_dict[...]` 直接下标（`:461`/`:483`/`:491`/`:494`），映射不到会 KeyError 而非静默丢弃。key 集合也逐个比过：官方 mtp 与导出件**双向差集为空**（各 2378）。

### 3.3 QuaRot 在运行时的真实作用

description 的元数据里有：

```json
"optional": {"quarot": {"rotation_map": {"global_rotation": "optional/quarot.safetensors"}}}
```

- `get_rotation_path()`（`models/llama_eagle3.py:31-41`）把这个相对路径拼到 **target 模型目录**（不是 draft 目录）。
- `get_rotation_matrix()` 只被 `llama_eagle3` 和 `kimi_k3` 调用，**DSV4 DSpark 从不加载这个矩阵**。
- 它在 DSpark 里唯一的作用是当**布尔量**：checkpoint 是 QuaRot 的 → 走 `dspark.py:436-439`，让 checkpoint 里 backbone 那个裸名 `embed.weight` / `head.weight` **不要**被灌进 drafter。
- `is_rot_weight_used()`（读 description 的 `is_rot_used`）全仓库只有 `deepseek_mtp.py:19-33` 用，它会额外建一个 `self.rot` 线性层在 forward 里转一次 hidden state；**DSpark 没有这个分支**。

所以：旋转是**静态折进权重**的，运行时不补偿。**导出时绝不能重跑 QuaRot。**

### 3.4 量化格式契约

- key 用原生命名（`mtp.0.attn.wq_a.weight`），不是 `self_attn`/`gate_proj`；
- 每个量化矩阵在 description 里占**三条**（`weight` / `weight_scale` / `weight_offset`），标签相同 → 2322 × 3 = 6966，加 56 条 FLOAT = 7022；
- `scale = absmax/127`，FP32 `[out,1]`；`offset` 全零 FP32；权重 clamp ±127；
- **FLOAT 张量的 dtype 是混合的**：`attn_sink` / `hc_*` / `head` / `markov_w2` / 各 norm / `gate.bias` 是 **F32**，`wo_a`（等）是 **BF16**。官方 56 条 FLOAT 里 44 个 F32、12 个 BF16。训练侧的 BF16 导出把 F32 那批压成了 bf16，导出器必须**逐 key 继承官方 dtype**；
- description 的五个元数据键必须保留：`version` / `model_quant_type` / `metadata` / `group_size` / `optional`。

### 3.5 msmodelslim PR #757 的两条

> "remove Flash mode MTP rotation entries, retaining only `mtp.0.main_proj` right-rotation to offset backbone Q-domain rotation"
> "copy the main model's unrotated embed and head to `mtp.0.embed` and `mtp.2.head`"

这解释了一个旧现象：官方 `mtp.0.embed` 与 target 的 `embed_tokens` 差 3.7、`mtp.2.head` 与 `lm_head` 差 4.4——**target 的那两个是旋转过的，mtp 里的是未旋转副本**。commit `642c237`（"用 target 的 head 打分会让 warm start 的 drafter 表现得像随机初始化"）因此有了机理解释。

---

## 4. 仍然没有解释的：BF16 路径为什么废

排除掉的：

| 位置 | 量化路径 | 未量化路径 | 结论 |
| --- | --- | --- | --- |
| Linear 布局 | `w8a8_dynamic.py:138` 转置后喂 `npu_quant_matmul` | `ops/linear.py:92-103` 不转置，`unquantized_gemm` 就是 `F.linear` | 各自自洽 |
| MoE 专家权重 | `w8a8_dynamic.py:342-343` `transpose(1,2)` | `routed_experts.py:78-82` 同样 `transpose(1,2)` | 对称 |
| MoE router | `model.py:280` 恒为 `quant_config=None` | 同左 | 一样 |
| `lm_head` | `dspark.py:322-326` 不传 quant_config | 同左 | 一样 |
| `swiglu_limit` / `routed_scaling_factor` | 来自 `FusedMoEConfig`，由 decoder layer 用 target config 建 | 同左 | 与量化无关 |
| `rotation_path` | 见 §3.3 | — | 对我们的 key 名 inert |

`maybe_trans_nz` 只改内存格式不改数值。**没找到机制。** 这不影响路线（已绕开），但要给 vllm-ascend 报 bug 的话，最小复现就是：官方 `mtp.*` 反量化成 BF16 放进独立 draft 目录，不给 `"quantization"`，接受率从 55% 掉到 <1%。届时需要那次启动的完整日志。

---

## 5. 量化导出器

`scripts/export_dspark_ascend_w8a8.py`：把 BF16 导出目录重新打包成 ModelSlim W8A8 的独立 draft 目录。

```bash
python scripts/export_dspark_ascend_w8a8.py \
    --draft-dir    <BF16 导出目录> \
    --official-dir /sharenfs/DeepSeek-V4-Flash-0731-w8a8 \
    --output-dir   <新目录> \
    --verify
```

做的事：

1. 从官方 description 取每个 `mtp.*` 张量的标签，并**要求 key 集合与导出件完全一致**（不一致直接报错）；
2. `W8A8_DYNAMIC` 的矩阵按 §3.4 的约定量化，产出 `weight`(I8) / `weight_scale`(F32 `[out,1]`) / `weight_offset`(F32 全零)；
3. `FLOAT` 张量**逐 key 继承官方 dtype**；
4. 分片（默认 4 GiB，按**源张量**规划，保证 scale 不会和它的 weight 分到不同分片）、写 index、写 description（生成后与官方逐条比对，不一致直接报错）、写 `config.json`（用官方那份，只改 `architectures` 和 `n_mtp_layers`）；
5. `--verify` 把产物反量化回来与输入逐张量比，误差以 **LSB 为单位**报告，超过 1 LSB 返回非零退出码。

不做的事：**不重跑 QuaRot / awq / smooth-quant**（§3.3、§3.5）。

已知的保真度上限：导出件是 bf16，scale 要从已舍入的值重算，所以产物与官方 INT8 **不是逐位相同**——实测每个元素最多差 1 步，0.03%~2.6% 的元素会移动，scale 相对差 ≤ 1/127（正好是"该行最大值量化成 126 而不是 127"的情形）。

服务时：

```bash
--speculative-config "{\"method\":\"dspark\",\"model\":\"<新目录>\",\"quantization\":\"ascend\",\"num_speculative_tokens\":7,\"enforce_eager\":true}"
```

---

## 6. 下一步

1. **identity 自检**：官方权重反量化成 bf16 → 过一遍导出器 → 起服务跑 200 条基准。应落在 **55.4% ± 0.5**。这一步不过，导出器就有问题，不要往下走。
2. **微调版导出** → 同一基准 → 与 55.4% 比。**这是微调第一次被公平评测。**
3. 若低于基线，回训练侧：`learning_rate`（100 步 × 2e-4 对 mHC 这类小量级参数偏猛）、是否冻结 mHC、`router_bias_update_rate`（短期微调建议设 0，见 §2.2 的漂移数据——它没有造成这次的故障，但那个漂移量本身不合理）。
4. 若要给 vllm-ascend 报 BF16 那个 bug，按 §4 的最小复现准备材料。

---

## 7. 本文没有验证的

1. BF16 独立目录失效的机理（§4）。
2. 训练侧 step-1 接受率的原始日志已丢失，"60~70% 且 100 步未下降"来自用户回忆，未复核。它现在只是旁证，不是链条的一环。
3. 导出器只跑过单测和合成数据的端到端，**尚未在真实 checkpoint 上跑过**——第 6 节第 1 步就是它的首次实跑。
4. 容器里实际安装的 `vllm_ascend` 是否逐字等于 `dedbb34`。`DSpark draft model loaded: 77 params` 与静态推算逐一吻合，这是运行时的旁证，但不是全量比对。
