# DSpark 词表裁剪:从可行性分析到训练评测端到端适配

日期:2026-08-05
涉及仓库:
- `specforge_sgl0514/SpecForge` — 分支 `feat/dspark-vocab-mapping`(基线 `7712377`)
- `specforge_sgl0514/DeepSpec` — 分支 `dspark-vocab-eval`(基线 `8f660b3`)
- 参照实现:`job/speculators/speculators`(fork 自 `vllm-project/speculators`,HEAD `f49226e`)

前置文档:
- `my_docs/2026-07-28_DSpark_VocabMapping_三方对比分析.md` — 可行性与先例
- `my_docs/2026-08-04_DSpark_VocabMapping_实现开发文档.md` — 实现细节与 review 逐条处置
- `my_docs/2026-08-04_DSpark_VocabMapping_代码Review.md` — 独立 review 报告

本文是把上面三份串起来的全景总结,重点记录**做了什么判断、为什么这么判断、哪些结论后来被推翻**。

---

## 0. 三句话

1. SpecForge 的 DSpark 现在支持词表裁剪了。`configs/*.json` 顶层加一行 `"draft_vocab_size": 32000` 是唯一开关,离线训练会自己从 feature 里建映射,不需要重新生成 hidden states。
2. 实测显存:objective 段分配从 **5.874 GiB 降到 2.690 GiB(2.18×)**。这个数比原分析文档推算的 4.75× 小得多,原因在 §4.2。
3. 过程中最大的收获不是功能本身,而是**"未裁剪路径不变"这个承诺被证伪了三次**。三次都是差分测试查出来的,其中一次(resume contract)已经在真实 NPU 集群上炸过。

---

## 1. 问题的形状

DSpark 论文的原始设计是"共享冻结的 target lm_head + 全词表",Markov 头做低秩分解 `B = W1·W2`,`W1 ∈ R^{V×r}`、`W2 ∈ R^{r×V}`,参数量随 V 线性增长。SpecForge 忠实实现了这一点,所以:

- `AlgorithmCapabilities.supports_vocab_mapping` 对 DSpark 是 `False`,而且这条契约被测试保护着
- `DSparkDraftModel` 继承链完全不经过 `Eagle3DraftModel`,连 `t2d`/`d2t` buffer 都没有
- 训练目标在全词表上做 CE,`objective_chunk_blocks` 分块存在的唯一理由就是 `[B,N,K,V]` 这个张量太大

而 speculators 从 DSpark 的第一版实现(#677)就带着词表裁剪,`MarkovHead` 一诞生就是 `verifier_vocab_size` / `draft_vocab_size` 双参数。已发布的 `novita/kimi-k2.6-dspark` 就是 163840→32000 的 5.1× 裁剪,跑在生产里。

**所以这不是"要不要做"的问题,是"SpecForge 缺了上游已有的能力"。**

### 1.1 speculators 穿刺出来的关键设计

对着 speculators 读了一遍,有两点是自己想不一定能想到的:

**其一,W1 和 W2 不共享词表。** `W1` 由**上一个真实 token** 索引,那个 token 可能是任意 target id,所以必须保持全词表;`W2` 输出的偏置要叠到 draft logits 上,所以跟随裁剪。SpecForge 原本两边都用 `self.vocab_size`,没有这个区分。这是设计上真正"想清楚了"的一点,后来 SpecForge 和 DeepSpec 两边我都是照这个形状改的。

**其二,词表裁剪不会让小模型"学得更难",反而更容易。** 因为教师和学生被施加了完全相同的限制——`lm_head` 和 `verifier_lm_head` 是同一份 target lm_head 按同一个 `t2d` 切出来的行,损失在 K 维上 softmax,意味着教师分布也被限制并重归一化了。学生要拟合的是 `p|_K`,比全词表目标更简单,去掉了长尾上那部分几乎学不出来也几乎采不到的质量。

真正的代价只有一个,而且是硬上界:**任何落在 draft 词表外的 target token,draft 永远提不出来,那个位置必然被拒。** 所以接受率天花板 ≈ 训练分布下 top-K 的覆盖率。这也是为什么后面要把 `draft_vocab_coverage` 做成训练指标——它必须出现在日志里,而不是事后审计。

---

## 2. SpecForge 侧实现

分支 `feat/dspark-vocab-mapping`,基线 `7712377`,7 个 commit,`26 files changed, +2796 −95`。

### 2.1 分层

| 层 | 改动 |
|---|---|
| 模型 | 新增 `modeling/draft/vocab_mixin.py`:`DraftVocabMappingMixin`,拥有 `t2d`/`d2t` 的注册、安装、校验与派生查找表。`Eagle3DraftModel` 也混入它,删掉重复的 `load_vocab_mapping` |
| 模型 | `DFlashDraftModel` 混入 mixin,新增 `draft_logits_from_target_head` / `draft_ids_to_target_ids` |
| 模型 | `VanillaMarkovHead` 的 `markov_w1` 保持 `vocab_size`,`markov_w2` 改用 `draft_vocab_size`;三种 head 类型都透传 |
| 目标 | `_dspark_objective_chunk_terms` 用按 `t2d` 切行的头;硬标签映射到 draft 空间,越界的记 `OUT_OF_DRAFT_VOCAB_LABEL` |
| 契约 | `dspark/providers.py` 补 `supports_vocab_mapping=True` + `vocab_mapping_modes={OFFLINE, STREAMING}` |
| 校验 | `planning.py` 新增两条:算法不支持 mapping 却给了路径 → 拒;支持但本次不裁剪却给了路径 → 拒 |
| 配置 | 新增 `configs/qwen3-8b-dspark-draftvocab32k.json`,**不改任何现存配置** |

### 2.2 三个值得记录的决定

**惰性 + 版本号缓存,而不是构造时切头。** 训练模型在 `build_model_bundle` 里就构造好了,但 t2d/d2t 要到 `_ensure_offline_vocab_mapping` 才装上。所以裁剪头必须首次使用时才切,并按 draft 模型的 `vocab_mapping_version` 做缓存键,后装的映射才能生效而不是继续用旧切片。

**`vocab_mapping_loaded` 做成从 buffer 派生的 property,而不是安装时置位的标志。** 这是 review 的 F1:buffer 也会经由 `load_state_dict` / `from_pretrained` 进来,而那两条路径不调用我们的任何方法。做成标志位的话,一个正确保存的裁剪 checkpoint 重新加载后会报告"没装映射"然后拒绝运行。改成 `bool(t2d.any())` 就一劳永逸——全零的初始 buffer 正是"还没装"的无歧义状态。

**不裁剪时不注册 buffer。** 无条件注册会给所有存量全词表 checkpoint 的 state dict 加上两个键,而 warm start 把任何缺失键都当致命错误。藏在 `use_draft_vocab` 后面,存量 checkpoint 逐字节可加载。

### 2.3 与 speculators 的一处实质差异

speculators 是纯软分布蒸馏(KL/TV/CE 都在 K 维 softmax 上),SpecForge 的 DSpark 是**混合目标**:硬标签 CE(α=0.1)+ 软分布 L1(α=0.9)+ confidence BCE。

硬标签这条 speculators 没有,所以裁剪时多一道工序:`target_ids` 是 target 词表的真实 token id,必须经 `t2d` 映射到 draft 空间,落在词表外的要排除。**这一步不能省,否则会出现"标签 id 越界 / 指向错误行"的静默 bug。**

而且 CE 拿到了自己的分母:被裁的标签同时退出分子和分母,否则 `alpha_ce` 会随覆盖率漂移。L1 项继续用全部监督位置——它的教师在每个位置都有定义,与标签是否被裁无关。

---

## 3. Review 与六条 finding

实现完成后请了一个独立 agent 做 review,报告见 `2026-08-04_DSpark_VocabMapping_代码Review.md`。**六条 finding 全部核实为真,无误报。** 逐条处置见实现文档 §6,这里只记最值得说的两条。

### 3.1 F4:我写的回归测试是空的

`test_identity_vocab_reproduces_the_unpruned_objective` 用同一 seed 在**同一分支上**建两个模型互比,只证明了初始化可复现,不能证明新实现与 `7712377` 的旧目标一致。

**我在开发文档里写的"unpruned 逐位不变、有测试守着"这个结论当时并不成立。** 这是最有价值的一条——它提供的是虚假的安全感,比没有测试更糟。

修法是在测试里**逐字转写** `7712377` 的旧公式作为参照实现,按名字逐项 bit-exact 比对。参照函数是冻结的:它由实现生成就检测不出实现的变化。

从这条开始,后面每一个测试我都做**变异验证**——注入对应故障,确认测试真的会红。不做这一步就不能说"有测试守着"。

### 3.2 F2:接受概率高估

裁剪后教师分布在 K 维上重新归一化,`accept_probability = 1 - 0.5·L1(q, p̃)`。但真实接受率是 `Σ_K min(q_i, p_i)`,用的是**未归一化**的 `p_i`。因为 `p̃_i = p_i/Σ_K p ≥ p_i`,当前值系统性偏高。

review 给的反例:target 在 4 个 token 上均匀、K 只保留 2 个、draft 恰好等于条件分布,当前代码算出 1.0,真值 0.5。

关键在于**未裁剪时这个量本来就是精确 speculative decoding 的接受率**,所以裁剪静默改变了它的语义,是回归而不只是不精确。

修法是把两个教师分开,合并任何一边都是错的:

| 用途 | 教师 | 理由 |
|---|---|---|
| L1 蒸馏损失(α=0.9) | `p̃`,在 K 上重归一化 | draft 物理上无法在 K 外放质量,其分布在 K 上必然和为 1。目标若改用和为 `m<1` 的真实 `p`,损失会有 `1-m` 的不可约下界并整体压偏梯度 |
| `accept_probability` | 真实 `p`,和为 `m ≤ 1` | 它是对服务端接受率的**预测**(喂 confidence head 的 BCE 目标和 `tau_probabilistic`)。被裁掉的概率质量在服务时是真实的拒绝 |

全词表归一化因子用 `_full_vocab_logsumexp()` 分块求:按词表走 `lm_head.weight`,每块立即 `logsumexp` 归约成标量,再用可结合性合并。**裁剪要避免的 `[..., V]` 张量始终没被物化**,调用点在 `no_grad` 内,不产生激活显存和反传。

顺带新增指标 `teacher_kept_mass`。**它界的是单步接受概率**(`accept_probability_j ≤ kept_mass_j`),**不是 `tau_probabilistic` 的上界**——后者是 `1 + Σ_j Π_{k≤j} accept_k`,含 anchor token 和累积乘积。这个说法我第一版写错了,后来纠正。

---

## 4. 三次"未裁剪路径不变"的证伪

上线前提是:不开小词表时,行为与 `7712377` 完全一致。我一开始基于单元测试就下了这个结论,**事实证明三次都不成立**。

### 4.1 差分验证方法

同一脚本在两个 worktree(HEAD / `7712377`)各跑一次,对 loss、accuracy、全部 ratio metrics 的分子分母、以及全部梯度做 sha256 逐位比对,覆盖 4 种配置(默认 / `loss_decay_gamma=3` / alpha 重排 / `ce_alpha=0`)。

三处偏差,没有一处是单元测试能发现的:

**(1) resume contract 多了一个 key —— 已在真实集群上炸过。**

`trainer.py:329` 对"契约里有、checkpoint 里没有"的 key 是硬报错。无条件写入 `dspark_draft_vocab_size` 会让**所有本功能之前产出的 DSpark checkpoint 全部无法 resume**,而该字段在未裁剪时只可能等于 `vocab_size`。

这条在用户的 8 卡 NPU 集群上真实发生了:

```
ValueError: checkpoint .../qwen3.6-27b-dspark-offline-step10000 does not record
required algorithm resume semantic dspark_draft_vocab_size
```

排查后确认是集群上的代码停在修复(`c9474a7`)之前的版本。修法是仅在 `use_draft_vocab` 时写入。事后用同一份未裁剪配置比对两个版本的 contract,十个 key 一字不差。

**(2) `loss_decay_gamma` 非空时最终 loss 不逐位相同。**

把 loss 拆成两个分母分别相除后,`x/D + y/D` 与 `(x+y)/D` 在 D 为非整数值浮点时舍入不同。`gamma=None` 时 D 是整数值、除法精确,所以默认配置看不出来——第一次验证正好没覆盖到。未裁剪路径改回原来的单分母表达式。

**(3) 多卡下多发了一次 `all_reduce`。**

未裁剪时 CE 分母与 loss 分母完全相同,却仍额外归约一次。**各 rank 必须在 collective 的数量和顺序上一致,这不是性能问题而是正确性问题**;而我的等价性验证全是单进程跑的,正好看不到它。改为仅在裁剪时发第二次归约,`local_ce_den` 也直接复用 `local_loss_den`。

这条是用户 review 我的修改时指出的。

### 4.2 现在的保证与代价

```
四种配置下 HEAD vs 7712377:loss / accuracy / 全部 ratio metrics / 全部梯度 逐位一致
```

三条各配了测试并做了变异验证。`test_full_vocab_run_emits_one_denominator_all_reduce` 用 mock 分布式环境断言未裁剪发 1 次、裁剪发 2 次——这类"单进程观测不到"的性质必须显式钉住。

**实测显存**(CPU fp32,V=151936、K=32000、block=7、chunk=128 blocks=896 token):

| 配置 | objective 分配 | 相对 |
|---|---|---|
| 全词表 | 5.874 GiB | 1.00× |
| 裁剪 K=32000(含 F2 修复) | **2.690 GiB** | **2.18×** |
| 裁剪 K=32000(若不做 F2) | 2.477 GiB | 2.37× |

H=256 与 H=1024 两次测得比值都是 2.18×,说明该段由词表维张量主导、比值不随 hidden size 变化。

几点必须说清楚:

- **原分析文档里的 4.75× 是错的量纲。** 那是 `[B,N,K,V]` 单个 logits 张量的比值,不是 objective 段的整体比值。confidence head、markov 的 rank 维中间量、hidden 相关张量都不随词表缩小。
- **F2 的显存代价比预估小得多**,只多 0.21 GiB(+8.6%),因为分块后立即归约、`[...,V]` 从未物化。我此前"教师侧算力回到基线"的说法在显存上不成立,说重了。
- **新增固定开销**:裁剪头拷贝随 H 线性增长,H=256 时 31 MiB,H=1024 时 125 MiB,qwen3-8b 的 H=4096 推算约 500 MiB(fp32)/ 250 MiB(bf16)。且 F2 之后 `self.lm_head` 全量权重不能释放,算归一化因子要用。
- **参数侧是干净收益**:markov 头 77.8M → 47.1M(只有 W2 缩小),但这部分是推算,未实测。

---

## 5. DeepSpec 侧评测适配

分支 `dspark-vocab-eval`,基线 `8f660b3`,commit `02b8155`。

DeepSpec 完全没有词表裁剪概念:`lm_head` 和 `markov_w2` 都按 `config.vocab_size` 建,没有 `t2d`/`d2t`。裁剪 checkpoint 装进去会在 `markov_w2` 形状上直接挂掉。

改动四处,前两处与 SpecForge 侧同形:

1. `markov_head.py` — W1/W2 词表分离,`sample_block_tokens` 加 `to_target_ids` 回调
2. `qwen3/modeling.py` — `lm_head` 按 K 建、仅裁剪时注册 buffer、`initialize_embeddings_and_head` 改成 `tgt_head[t2d]` 切行、新增 id 映射
3. `draft_ops.py` — **读代码时发现的额外一处**:`verify_draft_tokens` 会显式校验 `draft_probs` 宽度必须等于 target 的,而且用 **target 空间的 id** 去 gather。所以要先在 K 维归一化、再散射回 V 维(裁剪外补 0)。少了这步会直接抛异常,绕过校验则会 gather 到错误的列
4. `eval_real_dspark.py` — 原来的 `assert lm_head.weight.shape == tgt_head.shape` 在裁剪下必挂;并补读 config 顶层的 `draft_vocab_size`

### 5.1 集成验证

用真实 propose→verify 路径在小模型上跑通,并做了两侧键集比对:

```
SpecForge 有、DeepSpec 无: []
DeepSpec 有、SpecForge 无: ['embed_tokens.weight', 'lm_head.weight']
同名但形状不符: 无
```

严格子集,唯二多出的正是脚本里 `TIED` 已容忍并从 target 填充的两个。

不裁剪路径同样确认原样:`use_draft_vocab=False`、buffer 不进 state dict、id 映射是恒等、显式设 `draft_vocab_size=vocab_size` 与不设时键集一致。

---

## 6. 怎么用

### 6.1 开关只有一个

draft config **顶层**加一行(不要放进 `dflash_config`):

```json
"vocab_size": 248320,
"draft_vocab_size": 32000
```

所有读它的地方——prepare 脚本的 `_resolve_draft_vocab_size`、模型层的 `register_draft_vocab_buffers`、`build_markov_head`、planning 的 `_prunes_vocabulary`——都从顶层取。同一份 config 在 prepare 和训练时各传一次。

### 6.2 现有 hidden states 可以直接复用

DSpark 的离线 feature 只有 `input_ids` / `loss_mask` / `hidden_states` / `target_last_hidden_states`,**全部与词表无关**。映射由 `count_effective_feature_tokens` 从 feature 里的 `input_ids` + `loss_mask` 现场统计。

**调整 K 不需要重跑 prepare**:缓存键是 `sha256(dataset_identity : target_vocab_size : draft_vocab_size)`,改 K 自动换键、自动重建。前提是 yaml 里**不要设** `model.vocab_mapping_path`——设了就走显式文件,跳过自动建图。

代价是要扫一遍所有 feature 文件。`.gz` 会整份解压,非 gz 走 mmap 很便宜,所以 prepare 时**别加 `--compress`**。

### 6.3 要盯的信号

| 信号 | 阶段 | 含义 |
|---|---|---|
| `top 32000 token frequency ratio: XX%` | prepare 建图 / 训练建图 | **接受率天花板**,<95% 说明切太狠 |
| `draft_vocab_coverage` | 训练日志 | 按 token 计数的覆盖率,裁剪路径生效的标志 |
| `teacher_kept_mass` | 训练日志 | 按概率计的教师质量存活比例 |

### 6.4 两条硬约束

- **小词表必须从头训。** `markov_w2` 输出维从 V 变成 K,形状对不上,接不了现有权重;裁剪模型还多两个 buffer。这也意味着裁剪运行 resume 全词表 checkpoint 会被拒——那是正确行为。
- **`--minimum-valid-tokens` 与相邻性是两回事。** 前者统计任意位置的可训练 token 总数,只有 `scripts/filter_trainable_conversations.py` 查相邻性,而 DSpark 的 anchor 恰恰要求两个相邻。

---

## 7. 尚未解决 / 未验证

按风险排序:

1. **SGLang 的 DSpark 推理内核是否消费 `d2t`——未确认,硬阻塞。** 训练端做完但服务端不认,裁剪权重上线即错 token id。`export/to_sglang.py` 第一行就 `if strategy != "eagle3": raise`,DFlash/DSpark 只能走 `--to hf`。
2. **confidence head 该建模哪种验证协议——先于本次改动就存在,与裁剪正交。** 当前 `Σ min(q,p)` 对应拒绝采样式 speculative decoding。若服务端用贪心(`1[argmax q == argmax p]`)或"与 target 一个采样比对"(`Σ q_i·p_i`),公式还要再对齐。仓库内部本来就有三套口径并存:`spec_generate` 用第三种,objective 的 `accuracy` 用 argmax 相等,`tau_probabilistic` 用第一种。
3. **NPU 上的布尔行索引未验证。** `_pruned_head_state` 的 `weight[mask]`、DeepSpec 的 `tgt_head[mask]` 和 `full[..., kept] = probs` 都是布尔索引/索引赋值,Ascend 可能不支持。`draft_vocab_index()` 已经规避(挪到 CPU 构建),这几处没有。
4. **多卡 / 多机未实测。** collective 序列有测试钉住,但没在真机跑过。`lm_head`/`embed_tokens` 在 [backend.py:173](specforge/training/backend.py:173) 被显式放进 FSDP `ignored_modules`、保持不分片,所以取全量权重是安全的——这是读代码确认,非实测。
5. **DeepSpec 侧只在 CPU 小模型上验过**,没跑过真实 27B。

**后续优化方向(未做)**:target 的全词表 log-normalizer 是每 token 一个标量,完全可以在 capture 阶段和 `target_last_hidden_states` 一起存下来(1 float vs 248320,带宽可忽略),那样 F2 引入的额外全词表投影可以完全省掉。没在本轮做,因为要改 capture contract、normalizer、collator 和离线存储格式,并会让已采集的 feature 失效。

---

## 8. 复盘:两条方法上的教训

**其一,"有测试守着"这句话需要证据。** F4 那个空测试是我自己写的,而且我在文档里据此下了"逐位不变"的结论。从那以后每个测试都做变异验证——注入对应故障、确认测试真的会红。三次证伪里有两次是这么查出来的。

**其二,单进程验证证明不了分布式性质。** 多出来的那次 `all_reduce`、resume contract 的新增 key,都是在单机单进程下完全看不见的。凡是涉及 collective 序列、checkpoint 契约、跨 rank 一致性的改动,要么用 mock 显式钉住,要么就承认没验证过——不能因为单元测试全绿就说"不变"。

这两条的共同点是:**结论的强度必须匹配证据的强度。** "在我跑过的检查下未检出差异"和"无 bug"是两句不同的话。

---

## 附:commit 清单

**SpecForge `feat/dspark-vocab-mapping`**(基线 `7712377`):

| commit | 内容 |
|---|---|
| `430619a` | 词表裁剪初版实现 |
| `f42dca1` | F1 reload / F3 校验 / F4 真回归测试 / F5 退化值 |
| `2b19e27` | F6 拒绝换了映射的 resume |
| `8e07850` | 记录 review 处置,撤回 F4 证伪的结论 |
| `125bb60` | F2 上报真实接受率 |
| `c9474a7` | 未裁剪路径逐位不变(三处偏差) |
| `6aba90d` | cherry-pick:丢弃截断后无法训练的对话 |

**DeepSpec `dspark-vocab-eval`**(基线 `8f660b3`):

| commit | 内容 |
|---|---|
| `02b8155` | 评测侧词表裁剪适配 |
