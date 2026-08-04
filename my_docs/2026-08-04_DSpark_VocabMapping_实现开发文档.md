# DSpark 词表裁剪(vocab mapping)在 SpecForge 的实现 — 开发文档

日期:2026-08-04
分支:`feat/dspark-vocab-mapping`
基线:`fix/offline-reader-lazy-assembly` @ `7712377`(*offline: list feature files at assembly, read tensors lazily*)
工作目录:独立 worktree `SpecForge-dspark-vocab/`(主 worktree 当时在 `feat/epoch-progress-reporting` 上有未提交改动,其中包含 `assembly.py`,为避免污染而隔离)

本文写给 review 的人。第 1 节说清楚**依据是什么**,第 2 节说清楚**改了什么、为什么**,第 3 节列出**我认为最该被质疑的点**,第 4 节是**没做的事和前置条件**。

---

## 0. 一句话

给 DSpark 加上 `draft_vocab_size` 支持:draft 在 target 词表的高频子集上预测,`t2d`/`d2t` 描述该子集。**默认行为完全不变**——不配 `draft_vocab_size` 的 DSpark 走原路径,checkpoint 逐字节兼容,loss 数值逐位相同。

---

## 1. 参考了什么

### 1.1 本地文档

- `SpecForge/my_docs/2026-07-28_DSpark_VocabMapping_三方对比分析.md` — 本次工作的直接起点。其结论骨架正确,但有三处判断在本次复核中被修正,见 §1.4。
- `speculators/my_docs/2026-07-22_dspark_speculators工作总结.md` — 该文档记录 `--draft-vocab-size` 是显存头号杠杆(仅作为背景,未直接影响实现)。

### 1.2 本地代码仓 A:SpecForge(被修改方)

复核过、并作为实现依据的位置:

| 文件 | 看的是什么 |
|---|---|
| `specforge/algorithms/common/dflash_family_model.py:883-1167`(改前行号) | DSpark 三项损失的真实构成;`prev_token_ids` 的来源与 id 空间 |
| `specforge/modeling/draft/dspark.py` | `VanillaMarkovHead`/`Gated`/`RNN` 的 W1/W2 尺寸;`sample_block_tokens` 的自回归回喂 |
| `specforge/modeling/draft/dflash.py` | `DFlashDraftModel` 无自有 lm_head,借用 target 的;`spec_generate` 采样边界 |
| `specforge/modeling/draft/base.py:193` | `Eagle3DraftModel.load_vocab_mapping` 的既有实现 |
| `specforge/modeling/draft/llama3_eagle.py:1696`、`peagle.py:235` | EAGLE3/P-EAGLE 无条件注册 `t2d`/`d2t` 的既有写法 |
| `specforge/core/compact_teacher.py:110-230` | 已有的 `t2d` 切头工具与 `validate_vocab_mapping_consistency`(本次直接复用,未重写) |
| `specforge/training/assembly.py:203,408-499,620` | `bundle.draft_vocab_size` 已是通用字段;离线自动建图只在 colocated 路径触发 |
| `specforge/training/model_loading.py:117,442-447` | `draft_vocab_size` 缺省填充;**warm start 把任何 missing key 当致命错误** |
| `specforge/application/planning.py:172-186` | disaggregated 强制 `vocab_mapping_path` 的既有检查 |
| `specforge/modeling/auto.py:112-114` | `draft_vocab_size` 缺省等于 `vocab_size` |
| `specforge/core/chunking.py:15-90` | `checkpointed_chunk_reduce` 的对齐/切片语义(决定新参数怎么传) |
| `specforge/training/strategies/base.py:456-512` | `DSparkTrainStrategy.checkpoint_state_filter` 只剥 `draft_model.` 前缀 → buffer 会自然入 checkpoint |
| `specforge/export/checkpoint_io.py:98-110`、`to_hf.py`、`to_sglang.py:34,67` | 导出侧对 `t2d`/`d2t` 的既有容忍度;SGLang 专用导出器仅支持 EAGLE3 |
| `patches/sglang/v0.5.14/spec-capture.patch` | 确认本仓库的 SGLang patch 只做**采集**,不含 DSpark 推理内核 |
| `configs/*-dspark.json`、`examples/configs/*-dspark-*.yaml` | 现存 DSpark 配置一律无 `draft_vocab_size`;offline 用 `local_colocated`,其余 `disaggregated` |

### 1.3 本地代码仓 B:speculators(参照实现,只读,未修改)

路径 `/Users/wuyidong/projects/github_projects/job/speculators/speculators` @ `f49226e`(fork 自 `vllm-project/speculators`)。

| 文件 | 借鉴了什么 |
|---|---|
| `src/speculators/model.py:20-125`(`DraftVocabMixin`) | **条件注册**(`use_draft_vocab = draft != verifier`)的设计;shape 校验的写法 |
| `src/speculators/models/dspark/model_definitions.py:12-59`(`MarkovHead`) | **W1 走 verifier 词表 / W2 走 draft 词表**的非对称拆分——本次实现的核心依据 |
| `src/speculators/models/dspark/core.py:100-183` | `prev_token_ids` 取自 `input_ids`(verifier 词表)的确认 |
| `src/speculators/models/dflash/core.py:349-357` | 教师 logits 由**同一份裁剪头**产生 → 教师分布在 K 维重归一化 |
| `src/speculators/models/metrics.py:79-149` | 确认其 `ce_loss` 用的是 `argmax(teacher_logits)`,**全程不碰真实 token id** |
| `src/speculators/convert/dflash/converter.py:138` | `draft_vocab_size=transformer_config["vocab_size"]` 硬编码,佐证 SpecForge 侧此前无裁剪产出 |

### 1.4 对参考文档的三处修正(重要,直接改变了方案)

复核代码后,`2026-07-28` 那份分析有三处需要更正:

1. **§4.4 说"SpecForge 用的是 hard-label CE,不像 speculators 那样是软分布蒸馏"——只对了一半。**
   SpecForge DSpark 的损失是 `0.1×CE(真实 id) + 0.9×L1(软分布) + 1.0×confidence_BCE`。占大头的 L1 项的教师就是 `self.lm_head(aligned_target_hidden)`,和 speculators 的 `verifier_lm_head` 同构,**裁剪下只要师生共用同一份切过的头就自动成立,不需要任何 id 重映射**。真正需要重映射的只有 α=0.1 的 CE 和 accuracy 指标。
   反过来,该文档**漏掉了真正的坑**:损失分母。见 §2.4。

2. **§4.5 说"`_ensure_offline_vocab_mapping` 已有逻辑可直接复用,零额外前置"——错,且会造成回归。**
   该函数只在 **colocated offline** 路径被调用(`assembly.py:620`,在 `build_disaggregated_run` 的 early-return 之后)。更关键的是 `planning._validate_vocab_mapping` 只看 `vocab_mapping_modes` 非空、不看是否真的裁剪。若照原方案直接给 dspark 声明该能力,**所有现存的、不裁词表的 disaggregated DSpark 配置会立刻在 validation 阶段失败**。本次实现额外修了这个检查(§2.5)。

3. **§4.3 说"给 `t2d`/`d2t` 注册 buffer"——无条件注册会炸掉所有存量 checkpoint。**
   `model_loading.py:442-447` 把任何 missing key 当致命错误(只有 embedding 可豁免,而 DSpark 走的 `_finish_registered_draft` 连这个豁免都没有)。因此采用 speculators 的条件注册。

### 1.5 远端资料

均来自 `2026-07-28` 那份分析的检索结果,本次**未重新联网核验**,仅作为设计取值(32K)的背景:

- DSpark 论文 arXiv:2607.05147 — 原论文共享冻结 target lm_head、全词表,Markov 头参数 `2Vr` 随 V 线性增长
- FR-Spec arXiv:2502.14856 — 频次排序裁剪 LM head 搜索空间,保证输出分布等价
- VocabTrim arXiv:2506.22694 — 明确承认限制词表会轻微降低接受率
- `novita/kimi-k2.6-dspark`(HF)— `draft_vocab_size=32000` / verifier `163840`,已发布的 5.1× 裁剪 DSpark 权重
- `RedHatAI/GLM-5.2-speculator.dspark-preview`、`siro1/glm-5.2-dspark-spec-v1`(HF)— 全词表配方
- Red Hat / vLLM speculators v0.5.0 博文 — DFlash 官方训练命令用 `--draft-vocab-size 8192`

> 结论沿用:32K 附近是安全区,8K 是悬崖边。新增的示例配置取 32000。

---

## 2. 改了什么

### 2.1 新增 `specforge/modeling/draft/vocab_mixin.py`(新文件)

`DraftVocabMappingMixin`,统一 `t2d`/`d2t` 的所有权:

- `register_draft_vocab_buffers(vocab_size, draft_vocab_size)` — **仅在 `draft != vocab` 时注册 buffer**。这是存量 checkpoint 兼容的关键。
- `install_vocab_mapping(t2d, d2t)` — 唯一的安装入口。校验 shape + 复用既有的 `validate_vocab_mapping_consistency`(要求 `nonzero(t2d) == d2t + arange`),然后 `vocab_mapping_version += 1`。
- `load_vocab_mapping(path)` — 文件入口,内部转调上面那个。
- `draft_vocab_index()` — 派生 `long[V]` 查表(draft id,或 `-100`)。按 version 缓存,**不是 buffer**(纯派生量,不该进 checkpoint)。
- `require_vocab_mapping()` — 未安装即使用时报错。零初始化的 buffer 是**静默错误**(全 False 的 t2d 切出空头,全 0 的 d2t 把每个 draft id 映到 0),必须让它响。

`OUT_OF_DRAFT_VOCAB_LABEL = -100`:同时是 `F.cross_entropy` 的 `ignore_index`,又不可能等于任何真实 draft id,所以被裁的标签既不参与监督、又天然算作 accuracy miss。

`Eagle3DraftModel` 也混入了该 mixin,并删除了它自己那份 `load_vocab_mapping`(`base.py:193-206`)。EAGLE3/P-EAGLE 仍在各自 `__init__` 里无条件注册 buffer,未改。

### 2.2 `dflash.py` / `dspark.py` — 模型层

- `DFlashDraftModel(DraftVocabMappingMixin, Qwen3PreTrainedModel)`,`__init__` 里调 `register_draft_vocab_buffers`(在 `_init_draft_head` 之前,因为建 Markov 头要用 `draft_vocab_size`)。
- 新增 `draft_logits_from_target_head(target, hidden)` 和 `draft_ids_to_target_ids(ids)`。前者未裁剪时**仍调用 head 模块本身**(见 §3.1),裁剪时才取 `.weight` 行切片。
- `VanillaMarkovHead.__init__` 增加 `draft_vocab_size`:`markov_w1 = Embedding(vocab_size, r)` 保持全词表,`markov_w2 = Linear(r, draft_vocab_size)` 跟随裁剪。`Gated`/`RNN` 透传。
- `sample_block_tokens` 增加 `to_target_ids` 回调,在**回喂 `prev_token_ids` 之前**把采样出的 draft id 映回 target id。

  > 这是本次最隐蔽的一处。原代码 `prev_token_ids = next_token_ids`;裁剪后 `next_token_ids` 是 draft id,而 `markov_w1` 期望 target id。因为 K < V,索引合法,**不会抛异常,只会静默算错**。

### 2.3 `dflash_family_model.py` — 目标层(核心)

新增 `_pruned_head_state()`:惰性构造 `(裁剪后的头权重[K,H], 标签查表[V])`,按 `draft.vocab_mapping_version` 缓存。

> **为什么必须惰性**:训练模型在 `build_model_bundle`(assembly:207)就构造完了,而词表映射在 `_ensure_offline_vocab_mapping`(assembly:620)才安装。**顺序是反的**,不能在 `__init__` 里切片。按 version 缓存则保证后续重新安装映射能生效,而不是静默用旧切片。

`_dspark_objective_chunk_terms` 的改动:

| 位置 | 改动 |
|---|---|
| `base_logits` | `self.lm_head(...)` → `self.apply_objective_head(...)`,裁剪时输出 `[B,N,K,32000]`,**显存和 FLOPs 同步降 4.75×** |
| CE 标签 | 传入的 `target_ids` 改为"目标层自己的 id 空间";加 `ignore_index=-100` |
| CE 权重 | 新增 `ce_weights` 参数(= `loss_weights × 在表内`) |
| 教师 logits | 同样走 `apply_objective_head` — **师生共用同一份裁剪头**,这是 §1.4 第 1 条论证成立的唯一前提 |
| accuracy | 不变。`predicted_ids == -100` 恒假,被裁的标签自动算 miss,accuracy 因此如实反映接受率天花板 |
| 新增返回项 | `ce_eval_den`、`ce_position_den` |

### 2.4 损失分母(§1.4 提到的"真正的坑")

裁剪把目标函数劈成两半:

- **L1 / confidence 比较的是分布**,在每个被监督位置都良定义 → 继续用 `loss_weights` / `global_loss_den`。
- **CE 需要可实现的标签**,被裁位置必须同时退出分子**和分母** → 用 `ce_weights` / `global_ce_den`。

如果只做 `ignore_index` 而不改分母,CE 项会被"在表覆盖率"静默缩放(≈0.99,且随数据漂移),等于 α_ce 变成了一个会飘的量。

新的表达式:

```python
loss = world_size * ( α_ce·ce_num/global_ce_den
                    + (α_l1·l1_num + α_conf·conf_num)/global_loss_den )
```

**不裁剪时 `global_ce_den == global_loss_den`,与改前的单分母形式代数等价**——有测试守着(`test_identity_vocab_reproduces_the_unpruned_objective`)。

新增指标 `draft_vocab_coverage`(在表 token 占被监督 token 的比例)。这是接受率的硬上界,应该出现在训练日志里,而不是事后审计。`ce_loss` 的分母改为 `local_ce_den`,`ce_position` 的分母改为 `ce_position_den`,让日志里的 CE 在裁剪/不裁剪之间可比。

### 2.5 契约层与校验层

- `algorithms/dspark/providers.py`:`supports_vocab_mapping=True` + `vocab_mapping_modes={OFFLINE, STREAMING}`。两者必须同时改,否则 `common/providers.py:673-678` 的一致性检查会抛错。`resume_contract` 增加 `dspark_draft_vocab_size`。
- `application/planning.py`:
  - 新增 `_prunes_vocabulary(cfg, algorithm)` — 读已解析的 draft config,判断**这次运行是否真的裁剪**。disaggregated 强制 `vocab_mapping_path` 的检查改为以此为前提(修 §1.4 第 2 条的回归)。解析失败时返回 `True`,即**保持改前的严格行为**。
  - 新增:`vocab_mapping_path` 非空但算法不支持 → 报错。原先是静默忽略。
- `algorithms/model_providers.py`:`_finish_registered_draft` 补 `_load_vocab_mapping`,顺序与 EAGLE3 一致(warm start 在前,显式映射在后 → 显式路径覆盖 checkpoint 里的)。
- `training/assembly.py`:`_install_dataset_vocab_mapping` 里手写的 `d2t.copy_/t2d.copy_/vocab_mapping_loaded=True` 三行改为 `install_vocab_mapping(t2d, d2t)`,让所有安装路径共用同一套校验。

### 2.6 配置

**没有改动任何现存配置**。新增 `configs/qwen3-8b-dspark-draftvocab32k.json`(= `qwen3-8b-dspark.json` + `"draft_vocab_size": 32000`)作为示例与测试夹具。

### 2.7 NPU 考虑

`draft_vocab_index()` 的掩码赋值在 **CPU 构造后搬到设备**,避免依赖 Ascend NPU 的布尔 index-put。`lm_head.weight[t2d]` 行切片保留在设备上,但只执行一次并缓存。`dflash_family_model.py` 顶部既有的 NPU 分支(关闭 flex_attention)未改动。

### 2.8 测试

新增 `tests/test_utils/test_dspark_vocab_mapping.py`(13 个用例,CPU,用真实 DSpark 模型而非 stub):

1. 全词表 draft 的 state dict 不含 `t2d`/`d2t`(存量 checkpoint 兼容)
2. W1 全词表 / W2 裁剪
3. 未安装映射即 forward → 报错
4. **`draft_vocab_size == vocab_size` 时 loss 与 accuracy 逐位复现**(回归守卫)
5. 裁剪路径可训练,W2 拿到非零梯度,`draft_vocab_coverage` 被上报
6. **CE 分母 < L1 分母,且等于覆盖率分子**(§2.4 的守卫)
7. 标签查表:保留 token → `0..K-1`,被裁 token → `-100`
8. `draft_ids_to_target_ids` 与 `nonzero(t2d)` 一致
9. 不一致的 `t2d`/`d2t` 被拒
10. 重新安装映射会让缓存的裁剪头失效
11-13. planning:全词表 disaggregated 不再要求 `vocab_mapping_path`;裁剪时要求;不支持的算法配了就报错

被改动的既有测试(4 处,均为夹具修正而非放宽断言):

- `tests/test_runtime/test_offline_vocab_mapping.py` — 测试替身 `_DraftWithMapping` 改为直接使用生产 mixin,不再自己手写 `load_vocab_mapping`(**更忠实**,现在真正走到被测代码)
- `tests/test_config/test_schema.py`、`test_unified_feature_reachability.py` — 这些夹具把 EAGLE3 的 `vocab_mapping_path` 复用给了 dflash;新检查正确地拒绝了它,所以修夹具
- `tests/test_algorithms/test_builtin_providers.py` — resume contract 的 SimpleNamespace 替身补上 `draft_vocab_size`/`vocab_size`,并把新 key 加入期望集合

### 2.9 测试结果

```
python3.11 -m pytest tests/ -q --continue-on-collection-errors
基线(7712377):  50 failed, 865 passed, 26 skipped, 1 xfailed, 11 errors, 647 subtests passed
本分支:          50 failed, 878 passed, 26 skipped, 1 xfailed, 11 errors, 647 subtests passed
```

failed / SUBFAILED / ERROR 的**具体清单与基线逐行相同**(全部是本机无 CUDA、以及 py3.9 环境下的 collection error,与本改动无关)。passed +13 即新增用例。

格式化:`black==24.10.0` + `isort==5.13.2`(与 `.pre-commit-config.yaml` 中的 pin 一致)。

---

## 3. 我认为最该被质疑的地方

按我自己的怀疑程度排序,建议 review 重点看这几处:

1. **`apply_objective_head` 的两条分支不对称。** 未裁剪时调 `self.lm_head(x)`(模块),裁剪时走 `F.linear(x, weight)`。这不是洁癖问题——`tests/test_utils/test_dflash_losses.py` 注入了非 `nn.Linear` 的假 head(`_DualFixedHead`,靠调用顺序返回不同 logits)。我最初统一成 `F.linear` 直接打破了这个边界并挂掉 3 个测试。当前写法保住了未裁剪路径的模块语义,但代价是两条路径的 dtype 提升规则略有差异(裁剪分支多一次 `.to(weight.dtype)`)。bf16 训练下无差别,但值得确认。

2. **CE 与 L1 用了不同分母,是我的判断,不是唯一解。** 另一种做法是把"在表"掩码折进共享的 `loss_weights`,但那会让 L1/confidence 也丢掉被裁位置的监督——而那些位置的软目标是完全良定义的。我选了分开。若认为 α_ce 应当随覆盖率缩放,这里需要改。

3. **被裁位置没有截断整个 block。** 我只做逐点掩码,没有在 `eval_mask` 的 cumprod 之前折进"在表"判断。语义上,serving 遇到无法提议的 token 时后续必然全拒,所以"cumprod 截断"其实更忠实于接受率;但那会丢掉大量后续位置的监督。当前 `tau_probabilistic` 指标仍用原 `eval_mask`,因此**它没有反映词表裁剪带来的接受率损失**——`draft_vocab_coverage` 才反映。这个取舍需要确认。

4. **`_prunes_vocabulary` 在 validation 阶段解析 draft config,引入了新的 I/O。** 本地 JSON 无所谓,但 `draft_model_config` 为空、需要从 target 推导时可能触发 HF 下载。我用 try/except 包住并在失败时退回改前的严格行为,但这是新增的失败面。

5. **`install_vocab_mapping` 现在对 EAGLE3 也强制 `validate_vocab_mapping_consistency`。** 改前 `Eagle3DraftModel.load_vocab_mapping` 只是 `copy_`,不校验。这个不变量是 `d2t` 作为偏移表的正确性前提(`eagle3/model.py:164` 的 `pred + d2t[pred]` 依赖它),`process_token_dict_to_mappings` 的产物也满足它,所以我认为该收紧是对的。但**手工制作的历史映射文件可能会因此被拒**——这是本次唯一一处会影响 EAGLE3 既有用户的行为变化。

6. **FSDP 下的 `self.lm_head.weight`。** 裁剪头在首次 forward 时切片并缓存。切片产生的是新张量(拷贝),因此即便 FSDP 在 forward 后释放 gather 的分片,缓存仍有效。但我**没有在多卡环境实测过**,只在 CPU 单卡验证。这是最需要真机验证的一点。

---

## 4. 没做的事 / 前置条件

### 4.1 硬前置:SGLang 服务端是否消费 `d2t`(未解决)

`export/to_sglang.py:67` 第一行就 `if state.get("strategy") != "eagle3": raise`,DFlash/DSpark 只能走 `--to hf` + `scripts/gates/normalize_dflash_export.py`。而本仓库唯一的 SGLang 补丁 `patches/sglang/v0.5.14/spec-capture.patch` **只做采集,不含 DSpark 推理内核**。

**训练端做完、服务端不认 `d2t`,等于白做。** 这件事在真正启用裁剪训练之前必须先落实。本次改动不含 `normalize_dflash_export.py` 对 `draft_vocab_size` 的透传,原因是在确认服务端契约之前无法确定该写什么。

### 4.2 未做

- 未修改任何现存 `configs/*.json` 或 `examples/configs/*.yaml`。启用裁剪需要显式换配置。
- 未给 DFlash / Domino 启用词表裁剪。模型层的 buffer 与 `draft_logits_from_target_head` 对 DFlash 是通用的,但目标层 (`_dflash_objective_chunk_terms` / `_domino_objective_chunk_terms`) 未改,契约层也没开。所有新逻辑都用 `use_draft_vocab` 短路,Domino 路径不受影响。
- 未提供 online/disaggregated 场景下自动生成映射的工具。该拓扑仍要求显式 `model.vocab_mapping_path`(生产者与消费者无法各自推导出同一份映射),与 EAGLE3 现状一致。可用 `scripts/prepare_hidden_states.py` 或 `specforge/data/preprocessing.py::generate_vocab_mapping_file` 预生成。
- 未释放裁剪后 `self.lm_head` 的全量权重。qwen3-8b 上那是约 1.2 GB 的死重(裁剪头另占约 262 MB 拷贝)。可作为后续优化,但需要先确认没有别的使用者。
- 未做多卡 / NPU 实测(见 §3.6)。

### 4.3 预期收益(沿用参考文档的口径,未实测)

以 `qwen3-8b-dspark` (V=151936, K=32000, block=7, anchors=512, rank=256, chunk=128) 为例:

- objective logits 张量:每 chunk `128×7×151936×2B ≈ 272 MB` → `≈ 57 MB`,**4.75×**;同时头的 GEMM FLOPs 同比例下降
- Markov 头参数:`W1+W2 = 77.8M` → `47.1M`(只有 W2 缩小),对应 AdamW 状态省约 40%
- 代价:接受率上界 = 训练分布下 top-K 覆盖率,由新增的 `draft_vocab_coverage` 指标直接可观测

---

## 5. 变更文件清单

新增:
```
specforge/modeling/draft/vocab_mixin.py
configs/qwen3-8b-dspark-draftvocab32k.json
tests/test_utils/test_dspark_vocab_mapping.py
my_docs/2026-08-04_DSpark_VocabMapping_实现开发文档.md
```

修改:
```
specforge/modeling/draft/base.py                    删除重复的 load_vocab_mapping,混入 mixin
specforge/modeling/draft/dflash.py                  条件注册 buffer;裁剪头投影与 d2t 回映射
specforge/modeling/draft/dspark.py                  Markov 头 W1/W2 词表解耦;采样回喂修正
specforge/algorithms/common/dflash_family_model.py  惰性裁剪头;CE 标签重映射与独立分母;覆盖率指标
specforge/algorithms/dspark/providers.py            契约能力 + vocab_mapping_modes + resume key
specforge/algorithms/model_providers.py             registered draft 也加载显式映射
specforge/application/planning.py                   按"是否真的裁剪"判定;拒绝无效的 vocab_mapping_path
specforge/training/assembly.py                      统一走 install_vocab_mapping
tests/test_runtime/test_offline_vocab_mapping.py    测试替身改用生产 mixin
tests/test_config/test_schema.py                    夹具:不支持映射的算法不带 vocab_mapping_path
tests/test_config/test_unified_feature_reachability.py  同上
tests/test_algorithms/test_builtin_providers.py     resume contract 替身补字段 + 期望 key
```
