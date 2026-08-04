# DSpark Vocab Mapping 代码 Review

日期：2026-08-04  
Review 范围：`dfefcff`（`dspark: support draft vocabulary pruning (t2d/d2t)`）相对基线 `7712377`  
参考文档：`my_docs/2026-08-04_DSpark_VocabMapping_实现开发文档.md`

## 结论

建议 **修改后再合入**。

今天该分支只有上述 1 个新 commit。整体设计里，W1/W2 的词表空间拆分、采样后通过 `d2t` 回到 target id、CE 独立分母、惰性构建裁剪头等主干方向是合理的；相关单元测试也都能通过。

但 review 发现 2 个高优先级问题：

1. 裁剪模型写入 checkpoint/HF 目录后重新加载，`t2d/d2t` 虽然恢复了，运行态仍认为映射未安装，导致模型无法直接推理。
2. confidence head 的“接受概率”目标只比较裁剪后重新归一化的 K 维分布，没有计入 target 落在裁剪词表之外的概率质量；在标准 speculative decoding 语义下会系统性高估接受率。

此外还有 4 个中优先级的配置、resume 契约和测试覆盖问题。详细如下。

## Findings

### [P1] checkpoint/HF reload 不会恢复 `vocab_mapping_loaded`，导出的裁剪模型无法直接运行

位置：

- `specforge/modeling/draft/vocab_mixin.py:49-50`
- `specforge/modeling/draft/vocab_mixin.py:109-113`
- `specforge/modeling/draft/vocab_mixin.py:158-171`

`t2d`/`d2t` 是 buffer，会进入 `state_dict`；但 `vocab_mapping_loaded` 和 `vocab_mapping_version` 是普通 Python 属性，只会在 `install_vocab_mapping()` 中更新。标准 `load_state_dict()` / `from_pretrained()` 只恢复 buffer，不会调用 `install_vocab_mapping()`。

最小 round-trip 实测结果：

```text
saved_loaded True
reloaded_loaded False
buffers_equal True True
lookup_error RuntimeError ... no t2d/d2t mapping has been installed
```

因此，裁剪后的 DSpark 即便通过 `save_pretrained()` 完整保存，再通过 `AutoDraftModel.from_pretrained()` 加载，`draft_vocab_index()`、训练 objective 和 `spec_generate()` 仍会被 `require_vocab_mapping()` 拒绝。这与 `to_hf.py` 所声明的“SELF-CONTAINED、可直接 reload”契约冲突。

同一问题还会使“向已经执行过 forward 的模型重新 `load_state_dict`”无法递增 version、无法清理 `_draft_vocab_index` / `_pruned_head_cache`，存在继续使用旧派生缓存的风险。

建议：

- 给 mixin 增加 state-dict load hook，只有确认 `t2d`、`d2t` 均从 checkpoint 成功加载且通过一致性校验后，才设置 `vocab_mapping_loaded=True`、递增 version 并清空派生缓存。
- 增加 CPU HF round-trip 测试：安装非平凡映射 → `save_pretrained` → `from_pretrained` → 检查 mapping 状态并实际走一次裁剪投影/采样。
- exporter 的测试不要只比较 state dict，还要验证 reload 后模型可用。

### [P1] confidence/`tau_probabilistic` 使用条件分布，遗漏裁剪词表外的 target 概率质量

位置：

- `specforge/algorithms/common/dflash_family_model.py:976-995`
- `specforge/algorithms/common/dflash_family_model.py:1003-1017`
- `specforge/algorithms/common/dflash_family_model.py:1047-1052`

裁剪路径先只取 K 行 target head，再在 K 维上做 softmax：

```python
target_probabilities = softmax(target_logits_on_K)
accept_probability = 1 - 0.5 * L1(q_K, target_probabilities)
```

这得到的是 `p(token | token in K)`，不是 target 的完整分布 `p(token)` 在 K 上的原始概率质量。在标准 exact speculative decoding 中，draft 分布在 K 外为 0，真实单步接受率应是：

```text
sum(i in K) min(q_i, p_i)
```

而不是：

```text
sum(i in K) min(q_i, p_i / sum_K(p))
```

一个简单反例：target 在 4 个 token 上均匀分布，K 只保留前 2 个；draft 正好等于 K 上的条件分布 `[0.5, 0.5]`。当前代码得到 `accept_probability=1.0`，但对完整 target 分布的真实 overlap 只有 `0.5`。

影响：

- confidence head 默认参与训练（示例配置 alpha 为 1.0），会系统性学习到偏高的接受概率。
- `tau_probabilistic` 同样偏高；它不能反映词表裁剪造成的拒绝，这一点开发文档虽在“待质疑点”中提到，但 confidence loss 仍把该值当监督目标。
- 如果 serving 用 confidence 做动态 draft length/early stop，性能决策会失真。

条件 teacher 分布可以继续用于优化 K 内部的相对分布，但不能直接命名/使用为 serving 接受率。建议额外计算 target 的 kept mass（例如通过完整 log-normalizer 的分块计算），用未重新归一化的 `p_K` 计算 overlap；或者明确证明服务端采用了不同的条件验证协议，并补一条训练端与服务端公式一致性的契约测试。在 SGLang DSpark 裁剪契约尚未确认前，这一点应视为合入阻塞项。

### [P2] 全词表 DSpark 携带 `vocab_mapping_path` 时，planning 放行、模型构建却必然失败

位置：

- `specforge/application/planning.py:202-218`
- `specforge/algorithms/model_providers.py:86-94`
- `specforge/algorithms/model_providers.py:158-167`
- `specforge/modeling/draft/vocab_mixin.py:77-95`

planning 只在“算法完全不支持 mapping”时拒绝路径。DSpark 被静态声明为支持 mapping，因此全词表 DSpark（`draft_vocab_size == vocab_size`）带 `vocab_mapping_path` 会通过 validation。

但全词表 DSpark 为保持 checkpoint 兼容，不注册 `t2d/d2t`；随后 `_finish_registered_draft()` 无条件调用 `load_vocab_mapping()`，最终抛出：

```text
ValueError: t2d/d2t buffers are not present on this draft model; it was built without vocabulary pruning
```

该组合已做最小复现：planning 成功，mapping load 失败。

建议二选一：

- validation 在 `vocab_mapping_path` 非空但本次运行不裁剪时给出明确错误；或
- 像参照实现 speculators 一样，对全词表模型接受并忽略合法的 identity mapping。

同时增加“full-vocab + mapping path”的 planning/build 测试，避免 validation 与运行期契约分裂。

### [P2] 所谓“复现旧 objective”的回归测试实际只比较了两个相同的新模型

位置：`tests/test_utils/test_dspark_vocab_mapping.py:137-149`

`test_identity_vocab_reproduces_the_unpruned_objective` 用相同 seed 构造两次 `_build(VOCAB_SIZE)`，然后分别执行当前分支的同一套实现。两个模型相等只能证明初始化和随机 anchor 在重置 seed 后可重复，不能证明新实现与 `7712377` 的旧 objective 一致。

因此，开发文档中“loss 数值逐位相同，有测试守着”的结论目前没有被该测试实际保护。即使 `apply_objective_head`、分母或标签逻辑同时发生同样的回归，这个测试仍会通过。

建议：

- 在测试内保留一个直接按旧公式计算的 reference objective，使用同一份模型权重、同一批 anchor/输入与同一份 logits 比较；或
- 对固定小模型、固定输入保存经过人工核验的 golden loss/accuracy/各项 numerator-denominator。

### [P2] `draft_vocab_size=0` 被 `or vocab_size` 静默解释为全词表，且与 assembly 的值不一致

位置：

- `specforge/modeling/draft/vocab_mixin.py:66-69`
- `specforge/modeling/draft/dspark.py:309-317`
- `specforge/application/planning.py:195-199`

代码先执行：

```python
draft_vocab_size = int(draft_vocab_size or vocab_size)
```

所以显式配置 `0` 时会先被替换成 `vocab_size`，后面的 `<= 0` 校验永远捕获不到 0。最小实测得到：

```text
model_sizes 256 256 False False
```

但 `build_model_bundle()` 仍从原始 config 读取 `draft_vocab_size=0`。colocated offline 路径会据此尝试生成 K=0 的 mapping，再向“没有注册 buffer”的模型安装，最后以另一个无关错误失败。disaggregated planning 又会把 0 当成全词表，行为不一致。

建议仅把 `None` 当默认值，显式拒绝 0、负数、bool 和非整数，并增加 0/负数/大于 target vocab 的配置测试。

### [P2] resume contract 只记录 K，不记录 K 个 token 的身份

位置：`specforge/algorithms/dspark/providers.py:48-57`

新增的 `dspark_draft_vocab_size` 只能防止 K 改变，不能防止同样大小的 mapping 改变。注释所说的“防止 rows 的含义静默变化”并未由该字段实现。

当前生命周期中，显式/自动 mapping 在构建 trainer 前安装，而 `Trainer` 的 resume 随后从 checkpoint 再次 `load_state_dict`，会用 checkpoint 内的 `t2d/d2t` 覆盖本次配置的 mapping。同 K、不同 token 集合时不会报错，实际使用的 mapping 与本次配置路径/数据推导结果不一致。

建议把 mapping 的稳定 fingerprint 纳入 resume contract，或在 resume load 前后显式比较 checkpoint mapping 与本次 resolved mapping；不一致时应拒绝 resume，而不是静默覆盖。需要注意当前 `bind_runtime()` 发生在 offline 自动 mapping 安装之前，若要记录 fingerprint，安装/绑定时序也需要一起调整。

## 验证情况

执行了与本提交直接相关的测试集合：

```text
tests/test_utils/test_dspark_vocab_mapping.py
tests/test_utils/test_dflash_losses.py
tests/test_runtime/test_offline_vocab_mapping.py
tests/test_config/test_schema.py
tests/test_config/test_unified_feature_reachability.py
tests/test_algorithms/test_builtin_providers.py

结果：87 passed, 213 subtests passed
```

另外完成：

- `git diff --check 7712377 dfefcff`：通过。
- 本次修改的主要 Python 文件 `compileall`：通过。
- HF/save-pretrained mapping round-trip 最小复现：失败，见 Finding 1。
- full-vocab + mapping path 的 validation/build 契约最小复现：失败，见 Finding 3。
- `draft_vocab_size=0` 最小复现：模型层被静默当成全词表，见 Finding 5。

## 已确认合理的部分

- `markov_w1` 保持 target vocab、`markov_w2` 使用 draft vocab，符合 `prev_token_ids` 与输出 logits 的 id 空间。
- 自回归采样在回喂 W1 前执行 `d2t` 映射，避免 draft id 作为合法但错误的 target id 被静默使用。
- 被裁 CE 标签同时退出分子和 CE 分母，避免 `alpha_ce` 随 coverage 漂移；分布类 L1 项继续使用全监督位置，这个分母拆分本身是自洽的。
- 裁剪头按 mapping version 惰性缓存，解决了 model 构造早于 offline mapping 安装的时序问题；但 state-dict load 也必须纳入 version 生命周期，见 Finding 1。
- 不裁剪时不注册 `t2d/d2t`，确实避免给所有存量 DSpark checkpoint 新增 missing keys。

## 非阻塞观察与上线前置

- `load_vocab_mapping()` 使用 `torch.load(..., map_location="cpu")`，建议对纯 tensor mapping 改成 `weights_only=True`，减少不可信 pickle 的执行面。
- 开发文档已正确标出 FSDP 多卡、NPU 和 SGLang 服务端 `d2t` 消费契约尚未实测。这三项仍应作为真正启用裁剪训练/发布权重前的门禁；本次 CPU 单测通过不能替代它们。
