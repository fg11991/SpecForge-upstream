# DSpark 词表裁剪(vocab mapping)在 SpecForge 的实现 — 开发文档

日期:2026-08-04
分支:`feat/dspark-vocab-mapping`
基线:`fix/offline-reader-lazy-assembly` @ `7712377`(*offline: list feature files at assembly, read tensors lazily*)
工作目录:独立 worktree `SpecForge-dspark-vocab/`(主 worktree 当时在 `feat/epoch-progress-reporting` 上有未提交改动,其中包含 `assembly.py`,为避免污染而隔离)

本文写给 review 的人。第 1 节说清楚**依据是什么**,第 2 节说清楚**改了什么、为什么**,第 3 节列出**我认为最该被质疑的点**,第 4 节是**没做的事和前置条件**,第 6 节是**收到 review 后改了什么**。

> **本文已按 review 结果修订。** `my_docs/2026-08-04_DSpark_VocabMapping_代码Review.md` 提出 6 条 finding,全部核实为真。其中 F4 直接推翻了本文原先"unpruned 逐位不变、有测试守着"的结论——当时那个测试是空的。第 6 节记录了逐条处置,§2.8/§2.9 已同步更新。

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

**不裁剪时 `global_ce_den == global_loss_den`,与改前的单分母形式代数等价**——由 `test_full_vocab_objective_matches_the_pre_pruning_formula` 守着:该测试内联了一份 `7712377` 版目标函数的**逐字转写**,用同一份权重、同一批输入,对 14 个 numerator/denominator 逐项做 `torch.equal` 比对。

> 修订说明:此处原本引用的是 `test_identity_vocab_reproduces_the_unpruned_objective`,那个测试在同一分支上用同一 seed 建两个模型互比,只能证明可复现性,证明不了与旧目标函数一致(review F4)。现已替换,并通过注入变异验证过它会失败(见 §6.4)。

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
4. **`draft_vocab_size == vocab_size` 时,14 项 numerator/denominator 与 `7712377` 版公式逐位相同**(回归守卫,见 §2.4 修订说明)
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
python3.11 -m pytest tests/ -q --continue-on-collection-errors -p no:randomly
基线(7712377):  48 failed, 867 passed, 26 skipped, 1 xfailed, 11 errors, 652 subtests passed
本分支(2b19e27): 48 failed, 886 passed, 26 skipped, 1 xfailed, 11 errors, 677 subtests passed
```

failed / ERROR 的**具体清单与基线逐行相同**——用 `comm -23` 逐行 diff 过两边的 FAILED/ERROR 列表,新增为空。这些全部是本机环境所致(无 CUDA、无 `flash_attn`、无 `sglang`、macOS CPU 上的 flex_attention),与本改动无关。passed +19 即新增用例。

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
- 未释放裁剪后 `self.lm_head` 的全量权重(qwen3-8b 上约 1.2 GB,裁剪头另占约 262 MB 拷贝)。**注意:F2 修复后它不再是死重**——`_full_vocab_logsumexp` 需要全量行来算教师的归一化因子,所以除非改成在 capture 阶段预存该标量,否则不能释放。
- 未做多卡 / NPU 实测(见 §3.6)。

### 4.3 预期收益(沿用参考文档的口径,未实测)

以 `qwen3-8b-dspark` (V=151936, K=32000, block=7, anchors=512, rank=256, chunk=128) 为例:

- objective logits 张量:每 chunk `128×7×151936×2B ≈ 272 MB` → `≈ 57 MB`,**4.75×**
- 学生头 GEMM(含反传)同比例下降;**教师侧算力回到基线水平**,因为 §6.5 修 F2 时重新引入了一次全词表投影来算归一化因子。该投影分块归约、不物化 `[...,V]`,所以显存收益不受影响,算力收益打折。后续可通过在 capture 阶段存下 log-normalizer 标量彻底消除(见 §6.5)
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

---

## 6. Review 处置(2026-08-04,commit `f42dca1` / `2b19e27` / `HEAD`)

Review 文档:`my_docs/2026-08-04_DSpark_VocabMapping_代码Review.md`,针对 `430619a`。
**6 条 finding 全部核实为真,无误报,现已全部处置。** 其中 F1、F5 用最小脚本复现,F2 用推导确认,F3、F4、F6 读代码确认。

| # | 问题 | 处置 | commit |
|---|---|---|---|
| F1 | checkpoint reload 后 mapping 状态丢失,裁剪权重不可用 | 已修 | `f42dca1` |
| F2 | 接受概率遗漏裁剪词表外的概率质量,confidence 系统性偏高 | 已修 | `HEAD` |
| F3 | 全词表 + `vocab_mapping_path`:校验放行、建模必挂 | 已修(范围收窄) | `f42dca1` |
| F4 | "复现旧 objective"的回归测试实际是空的 | 已修 | `f42dca1` |
| F5 | `draft_vocab_size=0` 被静默当成全词表 | 已修 | `f42dca1` |
| F6 | resume contract 只记 K,不记 K 个 token 的身份 | 已修(换了落点) | `2b19e27` |

### 6.1 F1 — 改成从 buffer 派生,而不是加 load hook 补设标志位

Review 建议加 state-dict load hook,在确认 buffer 加载成功后再置 `vocab_mapping_loaded=True`。
实际采用的是更强的做法:**把 `vocab_mapping_loaded` 变成从 `t2d` 内容派生的 property**。理由是标志位的失效路径不止 `load_state_dict` 一条(还有 `from_pretrained`、直接 `copy_` 到 buffer、FSDP 的 buffer 广播),任何绕过我们方法的路径都会让标志位再次说谎;而"`t2d` 全 False"本来就是唯一无歧义的未安装态——真实映射必然恰好选中 `draft_vocab_size` 个 token。

派生缓存的失效仍然需要 hook,这部分按 review 的建议做了(`_invalidate_vocab_mapping_derivations` 挂在 load post hook 上,与 `install_vocab_mapping` 共用同一个失效点)。

### 6.2 F3 — review 的判断对,但我第一版把范围写宽了

按 review 原样实现后,`examples/configs/longcat-flash-eagle3-online.yaml` 立刻失败:它是全词表 EAGLE3(131072 == 131072)却带着 `vocab_mapping_path`。

关键区别:**EAGLE3 / P-EAGLE 无条件注册 `t2d`/`d2t`,所以这个组合今天装的是恒等映射,能正常跑**;DFlash 家族只在裁剪时注册,同样的配置才是硬失败。F3 描述的"校验放行、建模必挂"只对后者成立。用一条仓库里现存且能工作的配置去迁就我新发明的规则是本末倒置,因此把判据显式化为 `AlgorithmCapabilities.keeps_vocab_buffers_when_unpruned`,只对 DFlash 家族报错。

### 6.3 F4 — 新测试验证过"会红"

新的 `test_full_vocab_objective_matches_the_pre_pruning_formula` 内联 `_reference_dspark_chunk_terms`,是 `7712377` 版目标函数的逐字转写(刻意转写而非 import——从被测实现生成的参照检测不出被测实现的变化)。

为确认它不是又一个空测试,注入了两处变异并确认失败:

| 变异 | 结果 |
|---|---|
| `ce_num` 乘 1.001 | 1 failed |
| `accept_probability` 系数 0.5 → 0.4 | 3 failed |
| 还原 | 18 passed |

### 6.4 F6 — fingerprint 放不进 resume contract,改在 load 处比对

Review 已经指出时序问题:`bind_runtime()` 在 `build_model_bundle` 里执行,而 offline 自动 mapping 在 `_ensure_offline_vocab_mapping`(`assembly.py:607`)才安装,晚于前者。在 resume contract 里算 fingerprint 会记下一个空映射。

因此没有走 fingerprint,改为在 `load_state_dict` 的 pre hook 里直接比对:已装映射与 checkpoint 携带的映射不一致就报错。这个落点两份映射都在场,不依赖任何顺序假设。相同映射照常加载(正常 resume 路径)。`dspark_draft_vocab_size` 保留(能在加载任何权重之前拦下 K 变化),但注释不再宣称它能防住同 K 换 token。
### 6.5 F2 — 已修

Review 的数学是对的:`p̃_i = p_i / Σ_K p ≥ p_i`,所以 `Σ_K min(q_i, p̃_i) ≥ Σ_K min(q_i, p_i)`。原实现给出的是条件分布下的 overlap,系统性高于真实接受率。它给的反例(target 4 均匀、K 取 2、draft 恰为条件分布)算出 1.0 vs 真值 0.5,成立。**未裁剪时这个量本来就是精确 SD 接受率**,所以裁剪静默改变了它的语义,是回归而不只是不精确。

#### 关键区分:损失用条件分布,接受率用真实分布

这两个量必须分开,合并任何一边都是错的:

| 用途 | 教师 | 理由 |
|---|---|---|
| L1 蒸馏损失(α=0.9) | `p̃` = 在 K 上 softmax 重归一化 | draft 物理上无法在 K 外放质量,其分布在 K 上必然和为 1。若目标改用和为 `m<1` 的真实 `p`,损失会有 `1-m` 的不可约下界,并把梯度整体压偏。speculators 同样在 `draft_vocab_size` 维上 softmax |
| `accept_probability` | 真实 `p`(和为 `m ≤ 1`) | 它是对服务端接受率的**预测**,喂给 confidence head 的 BCE 目标和 `tau_probabilistic`。裁剪掉的概率质量在服务时是真实的拒绝 |

实现:`teacher_conditional = softmax(target_logits)` 供损失使用;`target_probabilities = exp(target_logits - full_log_normalizer)` 供接受率使用,后者是**未归一化的真实概率**。接受率改为 `Σ_K min(q_i, p_i)`,上界自然是 kept mass。

#### 全词表归一化因子怎么拿

新增 `OnlineDFlashModel._full_vocab_logsumexp()`:按词表分块走 `lm_head.weight`,每块立即 `logsumexp` 归约后只保留标量,最后用 `logsumexp` 的可结合性合并。**裁剪要避免的 `[..., V]` 张量始终没有被物化**;调用点在 `torch.no_grad()` 内,因此不产生激活显存和反传。

#### 代价(review 未估,这里说清楚)

多一次全词表投影。以 qwen3-8b(V=151936, H=4096)每 chunk 128×7 token 计,约 1.1 TFLOP,而裁剪后学生头的前反传合计约 0.7 TFLOP——**教师归一化会成为该段的主要开销**。

需要放到正确的坐标系里看:未裁剪的基线里教师本来就做一次全词表投影,所以这不是新增开销,而是**教师侧退回基线水平**。裁剪保住的是:

- 学生头前反传 4.75× 更省(大头,因为它有反传)
- `[B,N,K,V]` logits/CE/softmax 激活 4.75× 更省(原分析文档中显存的头号目标)
- 教师侧:显存仍省(分块归约,不物化),**算力回到基线**

**后续优化方向(未做)**:target 的全词表 log-normalizer 是每 token 一个标量,完全可以在 capture 阶段和 `target_last_hidden_states` 一起存下来,带宽代价可忽略(1 float vs 151936)。那样这次的额外投影可以完全省掉。没有在本轮做,因为它要改 capture contract、normalizer、collator 和离线存储格式,并会让已采集的 feature 失效——那是独立的一次改动。

#### 新增指标 `teacher_kept_mass`

裁剪时上报,`Σ_K p_i` 在被监督 token 上的均值,即**教师信念中能被裁剪词表触及的比例**。

它界的是**单步**接受概率:对第 j 个位置有 `accept_probability_j ≤ kept_mass_j`。**它不是 `tau_probabilistic` 的上界**——后者是 `1 + Σ_j Π_{k≤j} accept_probability_k`,含 anchor token 以及各步接受概率的累积乘积,和单步上界不是同一个量级的东西。做指标分析时不要把两者直接比。

与既有的 `draft_vocab_coverage` 是两个问题:后者答"实际出现的 token 有多大比例提得出来"(按 token 计数),前者答"教师的概率质量有多大比例活下来"(按概率计)。

#### 未裁剪路径保持逐位不变

未裁剪分支仍用原来的 `(1 - 0.5·l1).clamp(0,1)` 表达式,而不是统一成 `Σ min(q,p)`。两者在教师和为 1 时代数等价,但浮点算子序列不同、结果不逐位相同。§2.4 的参照测试要求 bit-exact,这个分支是刻意保留的——不能为了代码整齐给存量用户引入一个无收益的数值变化。

#### 测试

| 测试 | 保证 |
|---|---|
| `test_full_vocab_logsumexp_matches_the_dense_computation` | 分块归约 == 一次性 `logsumexp(lm_head(h))` |
| `test_full_vocab_logsumexp_is_chunk_boundary_independent` | chunk 取 7 / 64 / V / 2V 结果一致 |
| `test_acceptance_uses_true_target_mass_not_the_conditional` | review 反例的解析版:构造教师在 4 个 token 上均匀、K 只留 2 个,断言 `teacher_kept_mass ≈ 0.5`,并用 `Π_{k≤j} a_k ≤ a_j ≤ kept_mass` 得到的宽松上界 `1 + block·kept_mass` 兜住 tau |
| `test_full_vocab_run_reports_no_kept_mass` | 未裁剪时该指标不出现 |

变异验证(确认测试会红):

| 变异 | 结果 |
|---|---|
| `target_probabilities` 退回 `teacher_conditional`(重现 F2 原 bug) | `test_acceptance_uses_true_target_mass_not_the_conditional` failed |
| 分块归约只取首块 | `test_full_vocab_logsumexp_is_chunk_boundary_independent` failed |

#### 仍然存在的前置条件

§4.1 那条硬前置没有因此解除:**SGLang 的 DSpark 推理内核是否消费 `d2t` 仍未确认**。本次修的是"训练端报告的接受率与精确 speculative decoding 语义一致";如果服务端采用的是别的验证协议(例如贪心 argmax 匹配),该量的定义还需要再对齐一次。区别在于:现在的公式对应一个**明确且标准**的协议,而改前的公式不对应任何实际协议。
### 6.6 未裁剪路径的等价性(本轮补做的验证与修复)

上线前提是:**不开小词表时,行为与 `7712377` 完全一致**。本轮做了差分验证,并因此发现三处此前未被发现的偏差。

**验证方法**:同一脚本在两个 worktree(HEAD / `7712377`)各跑一次,对 loss、accuracy、全部 ratio metrics 的分子分母、以及全部梯度做 sha256 逐位比对;覆盖 4 种配置(默认 / `loss_decay_gamma=3` / alpha 重排 / `ce_alpha=0`)。

发现并修复:

1. **resume contract 多了一个 key,会让存量 checkpoint 无法续训。** `trainer.py:329` 对"契约里有、checkpoint 里没有"的 key 是硬报错。无条件写入 `dspark_draft_vocab_size` 会让**所有本功能之前产出的 DSpark checkpoint 全部无法 resume**——而该字段在未裁剪时只可能等于 `vocab_size`。改为仅在 `use_draft_vocab` 时写入。
2. **`loss_decay_gamma` 非空时最终 loss 不逐位相同。** 我把 loss 拆成两个分母分别相除,`x/D + y/D` 与 `(x+y)/D` 在 D 为非整数值浮点时舍入不同(gamma=None 时 D 是整数值,除法精确,所以默认配置看不出来)。未裁剪路径改回原来的单分母表达式。
3. **多卡下多发了一次 `all_reduce`。** 未裁剪时 CE 分母与 loss 分母完全相同,却仍额外归约一次 `global_ce_den`。各 rank 必须在 collective 的**数量和顺序**上一致,这不是性能问题而是正确性问题;单进程验证看不到它。改为仅在裁剪时发第二次归约,`local_ce_den` 也直接复用 `local_loss_den` 不再重算。

对应新增测试:

| 测试 | 保证 |
|---|---|
| `test_full_vocab_loss_keeps_the_single_denominator_form` | 未裁剪时最终 loss 逐位等于旧单分母公式,含 `loss_decay_gamma` 情形 |
| `test_full_vocab_run_emits_one_denominator_all_reduce` | mock 分布式环境,断言未裁剪发 1 次、裁剪发 2 次归约 |
| `DSparkResumeContractTest`(3 个) | 未裁剪的契约不含新 key;裁剪时含且只多这一个 key |

变异验证:三者分别注入对应故障后都会失败。

### 6.7 Review 没有覆盖到、我补跑的部分

Review 的验证清单里**没有任何 EAGLE3 路径的测试**,而本次改动动了 `Eagle3DraftModel` 的基类、删了它自己的 `load_vocab_mapping`、并让共用实现新增了旧路径从未做过的 `validate_vocab_mapping_consistency`(§3 第 5 条)。补跑:

```
tests/test_runtime/test_export.py
tests/test_runtime/test_checkpoint_resume.py
tests/test_runtime/test_no_sync_equiv.py
tests/test_runtime/test_equiv_4rank.py
→ 31 passed, 8 skipped,无回归
```

`tests/test_runtime/test_server_capture_gate.py` 的 collection error 是本机缺 `sglang`,已切到 `7712377` 对照确认基线同样失败。

另外 review 也没跑仓库级的配置清单测试,而 `430619a` 新增的 `configs/qwen3-8b-dspark-draftvocab32k.json` 漏了登记:`test_package_architecture.py`(dspark config 集合)、`test_example_draft_config_wiring.py`(每个 draft config 必须有 recipe)、`test_launch_topology.py`(每个 recipe 必须有 golden topology)三处都会失败。已补上登记,并新增 `examples/configs/qwen3-8b-dspark-draftvocab32k-offline.yaml`(选 offline/colocated,因为该拓扑能自行推导映射,不需要 `vocab_mapping_path`)。

### 6.8 未采纳的建议

- **`torch.load(..., weights_only=True)`**:torch ≥ 2.6 已是默认值,此处是 no-op。改前的 EAGLE3 代码用的是裸 `torch.load(file_path)`,现状不比它弱,未改。

### 6.9 Review 处置涉及的文件

```
新增:
examples/configs/qwen3-8b-dspark-draftvocab32k-offline.yaml   新 config 的 recipe(登记要求)

修改:
specforge/modeling/draft/vocab_mixin.py           F1 派生 property + 失效 hook;F5 严格校验;F6 冲突拦截
specforge/algorithms/contracts.py                 F3 新增 keeps_vocab_buffers_when_unpruned
specforge/algorithms/dspark/providers.py          F3 声明该能力为 False;F6 修正 resume key 的注释
specforge/application/planning.py                 F3 收窄后的校验
specforge/algorithms/common/dflash_family_model.py  F2 全词表归一化因子;真实接受率;teacher_kept_mass
tests/test_utils/test_dspark_vocab_mapping.py     F4 参照实现;F1/F2/F5/F6 新增用例(13 → 23)
tests/test_runtime/test_package_architecture.py   登记新 config
tests/test_config/test_example_draft_config_wiring.py  登记计数 63 → 64
tests/test_config/test_launch_topology.py         登记新 recipe 的 golden topology
tests/test_config/test_unified_feature_reachability.py  登记计数 63 → 64
```
