# DeepSeek-V4 多数据集离线准备与官方 encoder 适配

本文记录 2026-09-10 对 `prepare_hidden_states.py` 和 `deepseek-v4` 数据预处理的改动：
为什么要改、渲染和 loss mask 的最终契约、新的命令行用法，以及用真实 tokenizer 做的验证。

## 1. 改之前的问题

目标是把三份数据一起做离线特征：

| 文件 | 内容 | 希望的监督范围 |
|---|---|---|
| `cot.jsonl` | user → assistant（带 `reasoning_content`） | 所有 assistant 轮 |
| `nocot.jsonl` | user → assistant（无 reasoning） | 所有 assistant 轮 |
| `trace_final_answer.jsonl` | system(带 tools) → user → 多轮 tool call / tool 结果 → 最终回答 | 只有最后一个 assistant（前面各轮是别的模型写的） |

旧链路对这三份数据有四处问题，其中三处是静默的：

1. **trace 被整条丢掉。** 它用 OpenAI 的 `messages` 字段，`safe_conversations_generator`
   只读 `conversations`，于是得到空对话，被 `preprocess_conversations` 跳过。
2. **tools schema 丢失。** trace 的 tools 挂在 system 消息上，`GeneralParser` 只取
   system 的 `content`。
3. **tool call、tool 结果、reasoning 全丢。** DeepSeek-V4 的 checkpoint tokenizer 没有
   `chat_template`（`tokenizer_config.json` 里确实没有这个字段），所以总是走
   `GeneralParser` 的 fallback，而 fallback 只拼每条消息的 `content`。
4. **prepare 不能按文件区分监督范围。** 只接受一个 `--data-path`，也没有
   `--train-only-last-turn`（在线训练的 `data.train_only_last_turn` 有，离线准备没有）。

luwan233 的 [fg11991/specforge#1](https://github.com/fg11991/specforge/pull/1) 修了第 3 条里的
reasoning，但它的 fallback 仍然不渲染 tool call、tool 结果和 tools schema，历史轮也少了
官方格式里的 `</think>`。本次改动取代了它的 fallback，并保留它测试想守住的行为
（无 tools 的多轮只保留最后一轮 reasoning）。

## 2. 渲染：直接用官方 encoder

DeepSeek-V4 serving（SGLang）对每个请求都调用 `encoding_dsv4.encode_messages`。训练数据
用同一个 encoder 渲染，target 采出来的 hidden states 才和推理时的分布一致。

- `specforge/data/encoding_dsv4.py`：从 SGLang `python/sglang/srt/entrypoints/openai/encoding_dsv4.py`
  （commit `059269594`）原样拷贝编码部分，不含解码。以后从 serving 那份同步，不要在这里改。
- `DeepSeekV4Parser`（`parser_type="deepseek-v4"`）：做字段清洗后调用 encoder。
  - 只要有一个 assistant 带非空 `reasoning_content`，就用 thinking 模式，否则用 chat 模式。
    nocot 行里的 `think: true` 字段不参与判断。
  - system 消息上 JSON 字符串形式的 `tools` 会被解析回列表；行级 `tools` 按 serving 的做法
    挂到首条 system 消息上，没有 system 就插一条空的。
  - `tool` 消息由 encoder 合并成 `<｜User｜><tool_result>…</tool_result>`；`tool_calls` 渲染成 DSML 块。

reasoning 保留规则由 encoder 决定，和 serving 完全一致：

| 数据 | 渲染结果 |
|---|---|
| 单轮带 reasoning | `<bos><｜User｜>Q<｜Assistant｜><think>R</think>A<eos>` |
| 单轮无 reasoning | `<bos><｜User｜>Q<｜Assistant｜></think>A<eos>` |
| 多轮、无 tools | 最后一个 user 之前的 assistant 丢 reasoning：`…<｜Assistant｜></think>A1<eos><｜User｜>Q2<｜Assistant｜><think>R2</think>A2<eos>` |
| 声明了 tools | 每一轮 reasoning 都保留（encoder 看到 tools 就关掉 `drop_thinking`） |

## 3. loss mask 契约

每个 user 轮末尾由模板写入 `<｜Assistant｜>` 加一个 prompt 侧的 `<think>`（thinking）或
`</think>`（chat / 被丢弃 reasoning 的历史轮）。这个分隔符不是模型生成的，所以监督从它后面
开始，一直到 `<｜end▁of▁sentence｜>`（含）：

- 监督：reasoning 正文、模型自己输出的 `</think>`、answer、DSML 工具调用块、eos。
- 不监督：system（含 tools schema）、user、`<tool_result>`、prompt 侧的 `<think>` / `</think>`。

这里和旧模板（以及 PR #1）有一处刻意的不同：旧模板用 `ignore_token` 把所有 `<think>` /
`</think>` 都排除，模型在 reasoning 结束时生成的 `</think>` 也没有监督，drafter 在这个位置
学不到东西。现在只排除 prompt 侧那个。

`--train-only-last-turn` / `--last-turn-data-path` 只监督最后一个 assistant span。

## 4. `prepare_hidden_states.py` 新用法

```bash
torchrun --nproc_per_node=8 scripts/prepare_hidden_states.py \
  --strategy dspark \
  --target-model-path <DeepSeek-V4 checkpoint> \
  --draft-model-config configs/deepseek-v4-flash-dspark.json \
  --data-path cot.jsonl nocot.jsonl \
  --last-turn-data-path trace_final_answer.jsonl \
  --output-path ./cache/hidden_states/dsv4-mix-0910 \
  --chat-template deepseek-v4 \
  --max-length <覆盖 trace 长度> \
  --minimum-valid-tokens 10 \
  --tp-size 8 --batch-size 8 \
  --sglang-attention-backend ascend \
  --trust-remote-code
```

| 参数 | 含义 |
|---|---|
| `--data-path P [P ...]` | 监督每个 assistant 轮 |
| `--last-turn-data-path P [P ...]` | 只监督最后一个 assistant 轮 |
| `--train-only-last-turn` | 让 `--data-path` 的文件也只监督最后一轮（和训练配置 `data.train_only_last_turn` 同义） |
| `--shuffle-seed`（默认 42） | 每个文件内部 shuffle 和跨文件交错都用这个 seed |
| `--no-shuffle` | 按命令行顺序拼接，不交错（文件内部仍然 shuffle） |

流程：每个文件单独加载、单独分词（各自有 processed-dataset 缓存），打印每个文件的行数、保留
样本数、监督 token 数；然后合并、按 seed 交错、按 `--num-samples` 截断；vocab mapping 用合并后的数据生成。

需要注意的行为变化：

- **`--num-samples` 截的是合并后的数据集**，所以各文件按比例保留。代价是即使只要少量
  样本，也会先把所有文件分完词。原来无 filter 时是在分词前截取原始数据的前 N 行。
- **缓存 key 变了。** 新 key 包含文件路径、大小、mtime、监督模式、seed，以及 DSV4 渲染版本号
  （`_RENDERER_VERSIONS`）；所以第一次运行会重新分词，就地改过的文件也不会命中旧缓存。
- **`dataset_manifest.json`。** 输出目录里会记录数据来源、模式、seed、样本数。特征文件按
  样本下标命名，已存在的会被跳过，所以脚本拒绝往"manifest 不一致"或"有特征但没有 manifest"
  的目录里续跑，要求换一个新的 `--output-path`。
- 离线训练本身每个 epoch 还会再 shuffle（`launch.py`）。prepare 阶段交错的作用是让很长的
  trace 样本均匀分到各 DP rank，以及让 `--num-samples` 截到的是混合样本。

`scripts/filter_trainable_conversations.py` 也支持 `messages` 字段了。先过滤再准备时，trace
文件用 `--train-only-last-turn` 单独跑一次。

## 5. 真实 tokenizer 验证

用 HF `deepseek-ai/DeepSeek-V4-Flash` 的 `tokenizer.json` 在三份样例上验证：

| 数据 | 模式 | 和 SGLang encoder 逐字一致 | decode 还原 | 总 token | 监督 token |
|---|---|---|---|---|---|
| cot | 所有轮 | 是 | 是 | 2349 | 2333 |
| nocot | 所有轮 | 是 | 是 | 179 | 144 |
| trace | 只最后一轮 | 是 | 是 | 24451 | 3576（1 段） |
| trace | 所有轮（对照） | 是 | 是 | 24451 | 7490（4 段，前 3 段以 DSML 工具调用 + eos 结尾） |

`<think>`、`</think>`、`<｜Assistant｜>`、`<｜User｜>`、bos、eos 都是单个 token。

**trace 样本有 24451 token。** 截断是从右边截，而最后一轮在最末尾：`--max-length 4096`
会把要监督的最后一轮整个截掉，这条样本随即被 `--minimum-valid-tokens` / DSpark
loss-mask filter 过滤。`--max-length` 至少要覆盖 trace 的长度；每个文件打印的样本数能
直接看出被截掉了多少。

## 6. 代码位置

- `specforge/data/encoding_dsv4.py`：拷贝过来的官方 encoder（只含编码部分）
- `specforge/data/parse.py`：`DeepSeekV4Parser`；`GeneralParser._encode_with_loss_mask`（从 `parse` 里原样抽出来，行为不变）
- `specforge/data/template.py`：`deepseek-v4` 切到新 parser 和新 span 规则
- `specforge/utils.py`：`safe_conversations_generator` 支持 `messages`
- `scripts/prepare_hidden_states.py`：多数据源、交错、manifest
- `scripts/filter_trainable_conversations.py`：支持 `messages`
- 测试：`tests/test_data/test_deepseek_v4_parser.py`、`tests/test_scripts/test_prepare_hidden_states_sources.py`
