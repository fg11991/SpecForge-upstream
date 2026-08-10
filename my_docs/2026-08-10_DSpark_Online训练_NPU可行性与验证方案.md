# DSpark online 训练:代码事实、NPU 卡点与顺序验证方案

日期:2026-08-10
分析对象:`specforge_sgl0514/SpecForge` — 分支 `feat/dspark-vocab-mapping`(HEAD `18fbf3f`)
运行环境:昇腾 NPU(tige 平台 / 本地 docker),target = Qwen3.6-27B,strategy = `dspark`

前置文档:
- `my_docs/2026-08-05_DSpark_VocabMapping_端到端适配总结.md` — 词表裁剪端到端
- 仓库外:`DSPARK_NPU_PORT_PLAN_zh.md`(分支 `dspark-npu-offline`)— 当初**刻意只做 offline** 的决策记录

本文回答一个问题:**现在这套 NPU offline 流水线,要切成 online 需要什么、哪里会断、按什么顺序验证。**
本文是可行性与方案,**不包含任何已完成的代码改动**。

**实证基线**:本文所有关于外部依赖的论断,都在本地对 **sglang 源码树 `v0.5.14`(`49e384c`)** 和 **mooncake-transfer-engine aarch64 wheel(`0.3.12.post1` / cp311)** 做过实测,不是读文档推断。第一版草稿里列的四个"卡点"经实测**证伪了两个、重新定性了一个**,过程保留在 §3。

---

## 0. 三句话

1. 这个代码库里 **online ≡ disaggregated**,没有 colocated online 路径。切 online 的实质是:`data.hidden_states_path` 换成 `data.train_data_path`,加一段 `deployment.disaggregated`,然后**额外维护两个外部服务**(打过 patch 的 SGLang capture server + Mooncake)。prepare_hidden 脚本整个不再需要。
2. **唯一真实的工程量是 Mooncake。** PyPI 上 aarch64 wheel 是有的(cp311 齐全),但 `store.so` 和 `mooncake_master` 两个都**硬依赖 `DT_NEEDED: libcuda.so.1 + libcudart.so.12`** —— 昇腾节点上没有这两个库,`import mooncake.store` 会在 dlopen 阶段就死。解法是**从源码 `-DUSE_CUDA=OFF` 重新构建**(上游 CMake 有这个 option,还另有 `USE_ASCEND` 系列)。online 强制要 Mooncake(`launch_plan.py:270`),没有 `shared_dir` 退路。
3. 另外必须做的只有一件事:**不能用 `managed_local`**,因为它只写 `CUDA_VISIBLE_DEVICES`(`launch_plan.py:418/484/721/724`),NPU 上会让 capture server 和 trainer 抢同一批卡。走 external services 模式即可绕开,而且那条路径下 specforge 完全不碰设备可见性变量。

> 而 sglang patch 与 post-norm 这两件我原本担心的事,实测都不成立:patch 对 upstream v0.5.14 **干净可打**,upstream 0.5.14 **本身就有一等公民的 NPU 支持**;online 抓的 last hidden **确实是 post-norm**。详见 §3.2 / §3.4。

---

## 1. online 在这个代码库里的确切形状

`docs/basic_usage/disaggregated_training.md` 开篇即定死:

> 生产者采集或摄入 feature,消费者跑标准 trainer。**online 训练永远使用这个 producer/consumer 拓扑;不存在 colocated 的 target 推理路径,也不存在独立的 Python 训练入口。**

所以"online"在配置层面不是一个开关,而是**数据源的选择**(README 原文:"online/offline 模式由所选的 `data` 源决定,而不是文件名"):

| `data` 段填哪个 | 得到什么 |
|---|---|
| `hidden_states_path` | offline —— 消费预先落盘的 `.ckpt` |
| `train_data_path` | online —— 原始对话 JSONL 送给 capture producer |
| `prompts_path` | online —— 已 tokenize 的 JSONL(带 `input_ids` / `loss_mask`) |

三个进程组:

| 组件 | 职责 | external 模式下谁拉起 |
|---|---|---|
| **patched SGLang server** | 跑 target prefill,把 aux / last_hidden **直接写进 Mooncake**;`/generate` 响应里只回 key / shape / dtype | 你自己 |
| **Mooncake**(master + metadata) | feature 零拷贝中转 | 你自己 |
| **specforge producer** | 发 prompt、lease 调度、发布 SampleRef;**不加载任何 target 模型** | `specforge train` |
| **specforge consumer** | 就是现在的 trainer,DP 并行 | `specforge train` |

关键设计:**tensor 从不流经 producer 进程**(`inference/adapters/server_capture.py` 模块 docstring)。server 直接按 `MooncakeFeatureStore` 的 key 布局写入,producer 只从 `meta_info["spec_capture"]` 里拿元数据拼 `SampleRef`。

### 1.1 DSpark 用的是 DFlash 的采集契约

`algorithms/dspark/providers.py:188,204` —— `capture_method="dflash"`。online 的 feature 名和 offline **完全一致**:

```
aux_feature         = "hidden_states"
last_hidden_feature = "target_last_hidden_states"
passthrough         = input_ids, loss_mask
```

这一点很重要:说明 online 侧同样会索取 `target_last_hidden_states`,和我们在 offline 侧费劲修过的那个 post-norm last hidden(commit `ff8b621`,改用 forward hook 抓)是同一个语义位置。两条路径的实现完全不同 —— online 是 patch 在 `logits_processor` 里抓的,offline 是 hook —— 但 §3.4 已经把 online 这条的 norm 位置读实了。

---

## 2. 与现有两个脚本的对应关系

| 现在(offline) | 切 online 之后 |
|---|---|
| `qwen3_6_27b_prepare_hidden.sh`(torchrun 8 卡跑 `scripts/prepare_hidden_states.py`) | **整个删掉**。不再有落盘的 hidden 目录 |
| `HIDDEN_STATES_PATH=.../hidden`(几 TB) | 不存在。feature 在 Mooncake 里过一遍就被 `DPAckController` 回收 |
| 训练脚本里的 `data.hidden_states_path=...` | `data.train_data_path=/dpc/hot/.../qwen3_6_600k_distill_4096.jsonl` |
| 自己写 `torchrun ... -m specforge.cli train` | `specforge train -c cfg.yaml`(自己 self-launch torchrun);也可保留外部 torchrun,代码会检测并复用 |
| — | 新增:起 Mooncake、起 patched SGLang server |

`model` / `training` 段基本原样保留,包括 `vocab_mapping_path`、`num_anchors=1024`、以及 `f795823` 引入的梯度尖峰守卫(见附录 A)。

**收益**:省掉 hidden 落盘的磁盘与 IO,数据可以无限流式喂;
**代价**:多两个必须健康的外部服务,故障面显著变大。

---

## 3. 四个候选卡点,实测后剩下两个

第一版分析基于阅读 SpecForge 自身代码,列了四个卡点。随后把 sglang `v0.5.14` 源码树和 mooncake aarch64 wheel 都拉到本地实测,结论表:

| # | 第一版判断 | 实测结论 |
|---|---|---|
| 3.1 | `managed_local` 只写 CUDA 变量 | **成立**,但绕得开(不用 managed_local) |
| 3.2 | sglang patch 未必能打在 Ascend 版上 | **证伪** —— patch 对 upstream v0.5.14 干净可打,且 upstream 本身就支持 NPU |
| 3.3 | 不确定 aarch64 能不能装 mooncake | **重新定性** —— 能装,但 wheel 硬链 CUDA,**必须自行源码构建** |
| 3.4 | online 的 last_hidden 未必是 post-norm | **证伪** —— 读实了,确实是 post-norm |

### 3.1 `managed_local` 在 NPU 上不可用 —— 证据强度:**代码确证**

`launch_plan.py` 里托管栈设置设备可见性的三处,全部是 CUDA:

| 位置 | 内容 |
|---|---|
| `launch_plan.py:418` | mooncake service:`{"CUDA_VISIBLE_DEVICES": ""}` |
| `launch_plan.py:484` | capture server:`"CUDA_VISIBLE_DEVICES": ",".join(server.cuda_visible_devices)` |
| `launch_plan.py:721` | producer:`CUDA_VISIBLE_DEVICES = ""` |
| `launch_plan.py:724` | consumer:`",".join(managed_local.trainer_cuda_visible_devices)` |

NPU 上这个变量不生效,结果是 capture server 和 trainer 会**同时抢全部八张卡**,必炸。

**结论:必须走 external services 模式**(不写 `managed_local`,改写 `server_urls` + mooncake 端点)。
**附带的好消息**:这条路径下 specforge 完全不触碰设备可见性(上面四处都在 `if managed_local is not None` 分支内),所以 `ASCEND_RT_VISIBLE_DEVICES` 由你自己在每个进程的 shell 里控制,不会被覆盖。

### 3.2 SGLang patch —— 证据强度:**本地实测,已证伪为卡点**

- `pyproject.toml:26` pin 死 `sglang==0.5.14`
- patch 位于 `patches/sglang/v0.5.14/spec-capture.patch`(582 行),改 **11 个文件**:
  `server_args.py`、`layers/logits_processor.py`、`managers/{io_struct,schedule_batch,scheduler,detokenizer_manager,tokenizer_manager}.py`、
  `managers/scheduler_components/{batch_result_processor,output_streamer}.py`、`model_executor/model_runner.py`,
  外加**新增** `srt/spec_capture_sink.py`
- 应用脚本:`scripts/apply_sglang_spec_capture_patch.sh`(内容感知的幂等:把已应用的 patch 文本存在 `sglang/.spec_capture_patch.applied` 旁,patch 变了会先反向再重打)

**实测一:patch 干净可打。** clone `sgl-project/sglang` 的 `v0.5.14` tag(HEAD `49e384c`),`git apply --check -p1` 通过,零冲突。

**实测二(推翻了原判断):upstream sglang 0.5.14 本身就是支持昇腾的。** 不存在"要不要用 Ascend 分叉版"这个问题:

| 证据 | 位置 |
|---|---|
| `--device` 帮助文本列了 `npu` | `srt/server_args.py:798` |
| `ascend` 在 attention / sampling / lora / deterministic 各类后端候选里 | `server_args.py:116,205,209,211,216,283` |
| 专门的 NPU 后端处理入口 `_handle_npu_backends()` | `server_args.py:2513` |
| 完整的 Ascend PD-disaggregation 子包 | `srt/disaggregation/ascend/{conn,transfer_engine}.py` |

你们 prepare_hidden 一直在用 `--sglang-attention-backend ascend`,说明装的大概率就是 upstream 0.5.14。**所以 S0 的第三条检查预期是直接通过。**

**实测三:patch 是设备无关的。** 对整个 patch 文本 grep `cuda|nvidia|torch.cuda` 零命中;更强的证据是 sink 的取数路径 —— `batch_result_processor.py` 里新增的 `_append_spec_capture_states` 干的是:

```python
req.spec_capture_aux.append(logits_output.hidden_states[start:end].cpu().clone())
```

**`.cpu().clone()`** —— tensor 在进 Mooncake 之前就已经落到 host memory 了。所以"Mooncake 侧与加速器无关"从推断升级为确证。

### 3.3 Mooncake —— 证据强度:**本地实测,是唯一真实的工程量**

先说约束:`launch_plan.py:270` `raise ValueError("online disaggregated training requires Mooncake")`。`backend: shared_dir` 只在 offline 摄入时允许(`schema.py:426`)。**这是硬约束,没有退路。** 需要 `mooncake.store` Python 模块 + PATH 上的 `mooncake_master` 二进制(`launch_plan.py:902,908` 显式检查)。

**实测一:aarch64 wheel 存在,而且齐全。** PyPI `mooncake-transfer-engine` 最新 `0.3.12.post1`,207 个发布文件里 64 个是 aarch64;cp310–cp313 全覆盖(你们是 python 3.11.15,`cp311-manylinux_2_28_aarch64` 有)。解包后 `mooncake/` 下 `store.so`、`engine.so`、`mooncake_master`、`http_metadata_server.py` 全在 —— SpecForge 要的东西一件不缺。

**实测二(真正的问题在这):wheel 硬链 CUDA。** 解析两个关键二进制的 ELF `DT_NEEDED`:

| 文件 | 关键 NEEDED 项 |
|---|---|
| `mooncake/store.so`(SpecForge `import mooncake.store` 加载的就是它) | `libcuda.so.1`、`libcudart.so.12`、`libibverbs.so.1`、`libmlx5.so.1`、`libnuma.so.1`、`libcurl.so.4` |
| `mooncake/mooncake_master`(托管栈要拉起的 master) | 同上 |

`DT_NEEDED` 是**链接期硬依赖**,不是 dlopen 软加载。昇腾节点上没有 `libcuda.so.1` / `libcudart.so.12`,`import mooncake.store` 会直接在动态链接阶段失败,连 Python 代码都进不去。注意 `libibverbs`/`libmlx5` 也是硬依赖,即使你只想用 `protocol: tcp`。

**解法:从源码构建,关掉 CUDA。** 上游 `kvcache-ai/Mooncake` 的 `mooncake-common/common.cmake` 里:

```
option(USE_CUDA   "option for enabling gpu features for NVIDIA GPU" OFF)
option(USE_MUSA   ...)
option(USE_ASCEND "option for using npu with HCCL" OFF)
option(USE_ASCEND_DIRECT "option for using ascend npu with adxl engine" OFF)
option(USE_ASCEND_HETEROGENEOUS ...)
```

`USE_CUDA` 默认 OFF,是被检测逻辑自动打开的(`CMakeLists.txt:194,204,211`),官方 wheel 就是这么变成 CUDA 版的。**昇腾上应当 `-DUSE_CUDA=OFF` 构建**;既然 feature 已经在 host memory 里(§3.2 实测三),纯 CPU + TCP 构建就够用,`USE_ASCEND*` 那几个是给 NPU 直连 KV 传输用的,这个场景不需要。

> 顺带澄清一个容易走错的岔路:sglang 自己的 `srt/disaggregation/ascend/transfer_engine.py` 用的是 `memfabric_hybrid.TransferEngine`,那是**另一套 API**,不是 `mooncake.store`,不能拿来顶替。

### 3.4 `last_hidden` 的 norm 位置 —— 证据强度:**读实了,已证伪为风险**

offline 侧踩过两次坑(`6848871` 抓成 pre-norm、`ff8b621` 改 forward hook),所以对 online 这条同样存疑。实际把链路读通了:

patch 在 `logits_processor.py` 里新增:

```python
last_hidden_states_to_store = (
    hidden_states
    if (logits_metadata.capture_hidden_mode.is_full() and aux_hidden_states is not None)
    else None
)
```

这里的 `hidden_states` 就是 `LogitsProcessor.forward` 的入参。往上追一层,模型侧:

| 模型 | 最终 norm 施加位置 | 传给 logits_processor |
|---|---|---|
| `models/qwen2.py`(`Qwen3Model` 继承自 `Qwen2Model`) | `:391` `hidden_states = self.norm(hidden_states)` / `:393` `hidden_states, _ = self.norm(hidden_states, residual)`,**在 return 之前** | 是 |
| `models/qwen3_5.py`(**Qwen3.6-27B 走的就是这个**) | `:1284` / `:1286`,同一形状 | `qwen3.py:539` |

也就是说 `self.model(...)` 返回的已经是过完 final norm 的张量,`logits_processor` 拿到的、patch 存下来的,**就是 post-norm**。

S5 的数值比对仍然要做,但性质从"排障"降为"确认"。

---

## 4. 顺序验证方案

原则:**每一步有明确 pass 条件,过不了就停在那一步查,不要往下走**。S0–S5 在 docker 里做,S6 才上 tige。

### S0 · 可行性体检(10 分钟,不跑任何模型)

```bash
python -c "import sglang; print(sglang.__version__, sglang.__file__)"
```

```bash
python -c "import mooncake.store as s; print('mooncake OK', s.__file__)"; which mooncake_master
```

```bash
cd /sgl-workspace/SpecForge && SGL=$(python -c 'import sglang,os;print(os.path.dirname(os.path.dirname(sglang.__file__)))') && git -C "$SGL" apply --check -p2 patches/sglang/v0.5.14/spec-capture.patch && echo "PATCH APPLIES CLEAN"
```

按 §3 的实测,预期结果是:**第一条和第三条直接通过,第二条挂在 mooncake 上**。

- **第二条挂了(预期如此)** → 走 §3.3 的源码构建路线:

  ```bash
  git clone https://github.com/kvcache-ai/Mooncake.git && cd Mooncake && mkdir build && cd build && cmake .. -DUSE_CUDA=OFF -DWITH_STORE=ON && make -j && make install
  ```

  建完再回来重跑第二条。**这是整个 online 迁移唯一有实质工作量的环节**,建议单独排期。若 `libibverbs`/`libnuma` 也缺,一并 `yum/apt` 装上。
- **第三条挂了(不预期)** → 说明镜像里的 sglang 不是 upstream 0.5.14。看冲突落在哪:11 个文件里有实质逻辑的只有新增的 `spec_capture_sink.py`(必 clean)和 `logits_processor.py`,其余多是字段透传,手工对齐通常可做。

### S1 · 打 patch,确认 flag 被识别

```bash
bash /sgl-workspace/SpecForge/scripts/apply_sglang_spec_capture_patch.sh
```

```bash
python -m sglang.launch_server --help 2>&1 | grep -A2 "spec-capture"
```

**pass**:`--enable-spec-capture`、`--spec-capture-method`、`--spec-capture-aux-layer-ids` 三个都出现。这步纯 argparse,与 NPU 无关。

### S2 · 起 Mooncake

```bash
mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=127.0.0.1 --rpc_port=35551 --http_metadata_server_port=35880 --metrics_port=35903
```

```bash
curl -s "http://127.0.0.1:35880/metadata?key=specforge-health-check"; echo
```

**pass**:metadata 端点有响应,35551 端口 listen。

### S3 · 取 aux layer ids,起 capture server

层号必须让 specforge 自己算(`resolve_capture_layers` 从 draft config 推),不要手填:

```bash
cd /sgl-workspace/SpecForge && python -c "
from specforge.config import load_config
from specforge.application.composition import resolve_run
from specforge.training.capture_contract import resolve_server_capture_contract
cfg = load_config('/path/to/qwen3.6-27b-dspark-online.yaml')
print(resolve_server_capture_contract(cfg, algorithm=resolve_run(cfg).algorithm))
"
```

会打出 `method='dflash'`、`aux_layer_ids=(...)`、`target_hidden_size`、`target_vocab_size`、`draft_vocab_size`。填进下面:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 GDN_ATTN_BACKEND_TRITON=1 python -m sglang.launch_server --model-path /dpc/hot/model/y00830025/Qwen3___6-27B --dtype bfloat16 --trust-remote-code --skip-tokenizer-init --tp-size 4 --chunked-prefill-size -1 --disable-radix-cache --enable-spec-capture --spec-capture-method dflash --spec-capture-aux-layer-ids <填ids> --attention-backend ascend --context-length 4103 --mem-fraction-static 0.8 --host 127.0.0.1 --port 30000
```

不能改的几项:

| 参数 | 为什么 |
|---|---|
| `--skip-tokenizer-init` | producer 直接送 `input_ids`,server 不需要 tokenizer |
| `--chunked-prefill-size -1` | capture 拒绝 chunked prefill |
| `--context-length ≥ max_length + 7` | `SGLANG_CAPTURE_CONTEXT_HEADROOM = 7`(`config/schema.py:29`) |
| `GDN_ATTN_BACKEND_TRITON=1` | 沿用 prepare_hidden 的结论:mega_chunk_gdn C++ kernel 不支持 Qwen3.6-27B 在 TP=4/8 下的 per-rank head 配比 |

**pass**:`curl http://127.0.0.1:30000/health` 返回 200。
**这一步是整件事的命门** —— 打过 patch 的 sglang 能否在 Ascend 后端起 27B,决定后面全部。

### S4 · 只看计划,不启动进程

```bash
cd /sgl-workspace/SpecForge && specforge train -c /path/to/qwen3.6-27b-dspark-online.yaml --plan
```

**pass**:配置校验通过,role 拓扑与端点解析正确。几秒钟,能提前抓出全部 schema 错误(未知字段、`tp_size≠1`、缺 mooncake 端点等)。

### S5 · 极小 e2e —— 最关键的一步

```bash
cd /sgl-workspace/SpecForge && ASCEND_RT_VISIBLE_DEVICES=4 SPECFORGE_DEVICE=npu specforge train -c /path/to/qwen3.6-27b-dspark-online.yaml "data.max_prompts=8" "training.max_steps=2" "training.batch_size=1" "training.accumulation_steps=1" "deployment.trainer.nproc_per_node=1"
```

按顺序看四件事:

1. producer 起来,prompt 被 lease 出去;
2. server 日志出现 capture 请求,`meta_info["spec_capture"]` 回了 key / shape / dtype;
3. consumer 从 Mooncake `get` 到 tensor,loss 有限;
4. **数值比对(治 §3.4)**:同样这 8 条 prompt 用 `scripts/prepare_hidden_states.py` 再跑一遍,把 online 的 `hidden_states` / `target_last_hidden_states` 与 offline `.ckpt` 逐元素比,bf16 容差 2e-2。

> `tests/test_runtime/test_server_capture_gate.py` 是官方对这条链路的门禁(live server + live mooncake + 一步真实训练 + 与 HF `output_hidden_states=True` 比对),可以拿来当写比对脚本的参照;**但它头上是 `torch.cuda.is_available()` 门控的,NPU 上直接 skip**,不能指望它替你验证。

### S6 · 单节点全量 online

卡分配:capture server TP=4 吃 NPU 0–3,trainer DP=4 吃 4–7。

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 SPECFORGE_DEVICE=npu specforge train -c /path/to/qwen3.6-27b-dspark-online.yaml
```

先跑 200 步,盯三件事:

- **喂得饱吗**:capture 吞吐够不够 4 个 trainer rank。不够就在 `server_urls` 里**把同一个 URL 重复几遍** —— doc 明说这是刻意支持的语义,每个条目起一个 rollout worker,可在不加 target 副本的前提下提高 prefill 占用率。
- **背压**:`runtime.in_flight_high/low_watermark` 是不是一直贴着上限。
- **精度**:accuracy 曲线与 offline 同期是否一致。

### S7 · 上 tige

到这一步脚本形态才定得下来。要点:

- 起**两组**进程:一组 mooncake + sglang server,一组 `specforge train --role consumer`;单节点也可后台拉起前两个再跑 `--role both`。
- **producer 不能放进多 rank 的 torchrun 里** —— 代码显式拒绝,防止重复采集(`launch_plan.py` ~740)。
- `--role both` 拒绝 `nnodes > 1`:SpecForge 不会 SSH 到远端节点,多节点必须显式分 role。
- `control_dir` 每次 attempt 必须全新;`consumer_state_dir` 必须节点本地。

---

## 5. YAML 骨架

```yaml
# model 段直接沿用现在 offline 那份(target_model_path / draft_model_config /
# trust_remote_code / embedding_key / mask_token_id / torch_dtype /
# vocab_mapping_path 全保留)。
# 注意:external 模式下 model.sglang_* 全部是惰性的 —— 它们只被 managed_local
# 用来拼命令行。自己起 server 时必须写在命令行上。
model:
  target_model_path: /dpc/hot/model/y00830025/Qwen3___6-27B
  draft_model_config: /dpc/hot/w00958190/dspark/configs/qwen3.6-27b-dspark-vocab-64000.json
  target_backend: sglang
  trust_remote_code: true
  vocab_mapping_path: /dpc/hot/w00958190/dspark/configs/vocab_600k_64000/vocab_mapping.pt
  # embedding_key / mask_token_id / torch_dtype 按现有 offline yaml 填

data:
  # 唯一的模式开关:hidden_states_path -> train_data_path
  train_data_path: /dpc/hot/w00958190/dspark/training_data/qwen3_6_600k_distill_4096.jsonl
  max_length: 4096
  chat_template: qwen3.5
  cache_dir: /dpc/hot/w00958190/dspark/cache
  build_dataset_num_proc: 64

training:
  strategy: dspark
  num_epochs: 1
  batch_size: 1
  accumulation_steps: 128
  learning_rate: 0.0006
  warmup_ratio: 0.04
  max_grad_norm: 1.0
  grad_spike_skip: "on"        # 见附录 A;yaml 里 on 必须加引号
  grad_spike_ratio: 10.0
  adam_beta2: 0.95
  num_anchors: 1024
  loss_decay_gamma: 4.0
  objective_chunk_blocks: 128
  tp_size: 1                   # online consumer 必须为 1;target TP 配在 server 上
  save_interval: 125
  dist_timeout: 30

run_id: qwen3.6-27b-dspark-online-001
output_dir: /dpc/hot/w00958190/dspark/newly_specforge/qwen3.6_dspark/online-001/output

deployment:
  mode: disaggregated
  trainer:
    nnodes: 1
    nproc_per_node: 4
  disaggregated:
    control_dir: /dpc/hot/w00958190/dspark/newly_specforge/qwen3.6_dspark/online-001/control
    consumer_state_dir: /tmp/specforge/online-001/consumer-state   # 必须节点本地
    backend: mooncake
    server_urls:
      - http://127.0.0.1:30000
      # 喂不饱 trainer 时重复同一 URL,每条 = 一个 producer worker
    mooncake_metadata_server: http://127.0.0.1:35880/metadata
    mooncake_master_server_addr: 127.0.0.1:35551
    mooncake_protocol: tcp

runtime:
  producer_lease: 8
  producer_concurrency: 1
  in_flight_high_watermark: 256
  in_flight_low_watermark: 192
```

`total_steps` / `max_steps` 都省掉是**可以的**:producer 会按 prompt 数、`num_epochs` 和 dispatch quantum(`consumer_world_size × batch_size × accumulation_steps`)算出精确 horizon 发布给 consumer,consumer 验证该契约后训到 EOF。

Mooncake 端点也可以不写死在 YAML 里,改用环境变量注入(`MOONCAKE_METADATA_SERVER` / `MOONCAKE_MASTER_SERVER_ADDR` / `MOONCAKE_LOCAL_HOSTNAME`),环境值优先级高于 YAML —— 同一份 recipe 换节点跑时用这个。

---

## 6. 未验证项登记

§3 已经把外部依赖类的疑问都实测掉了,剩下的都是**只能在昇腾硬件上跑出来**的:

| # | 未验证项 | 什么时候有答案 | 错了怎么办 |
|---|---|---|---|
| 1 | `-DUSE_CUDA=OFF` 构建出的 mooncake 能否正常跑 store put/get | S0 之后、S2 | 无退路,online 强制要 mooncake。这是最该先做的一件事 |
| 2 | 打过 patch 的 sglang 在 Ascend 后端跑 27B 的稳定性 | S3 | 无退路,只能查 |
| 3 | capture 吞吐能否喂饱 DP=4 | S6 | 重复 `server_urls` 加 worker;或降 DP |
| 4 | online / offline 两条路的 hidden 数值一致性(§3.4 已读实代码,但没跑过) | S5 | 照 `ff8b621` 的思路查 |
| 5 | 本文引用的 SpecForge 行号来自 GitHub `feat/dspark-vocab-mapping`;实际运行的是 codehub `SpecForge-Ascend` 同名分支 | 随时 | 以 codehub 那份为准 |

### 两处文档漂移(照着找会浪费时间)

1. `docs/basic_usage/disaggregated_training.md` 提到的 `scripts/gates/run_disaggregated_overfit_gate.sh` **在本分支上不存在**(`scripts/gates/` 下只有 `_e2e_common.sh`、`normalize_dflash_export.py`、`run_dflash_chat_serving_gate.py`)。
2. `specforge/inference/sglang_patch_inventory.md` 称"采集启动配置因此保留 radix cache 开启",但 `launch_plan.py` 实际给 capture server 传的是 `--disable-radix-cache`。两者矛盾;本文 S3 的手工启动命令沿用**代码**的行为(关掉),因为那才是被测试覆盖的组合。

---

## 附录 A:`f795823` 梯度尖峰守卫

与 online 无关,但会一起写进上面的 YAML,一并记录。

commit `f795823` "optimizer: discard gradient-norm outliers instead of scaling them down"。动机:一次 27B DSpark 实跑在 10480 步内**四次**丢掉约 4000 步进度(accuracy 在十步内从 0.55 掉到 0.15,再花 60–820 步爬回)。四次 onset 的共同点只有一个:梯度范数 5.6–30.1,而健康上限是 1.5。

**为什么 `max_grad_norm` 治不了**:它只约束更新的**模长**,不约束**方向**;而 AdamW 的 per-parameter step 对梯度的全局缩放几乎免疫。被裁剪的离群步依然把每个参数推了约一个 lr,方向还是离群那个方向;更糟的是被污染的 `exp_avg_sq` 会把一次坏步拖成几百步的低效学习。

**做法**:整步丢弃 —— 梯度清空(`p.grad = None`,而非 `zero_grad` 后照常 step,因为后者仍会衰减动量并施加解耦权重衰减)、Adam 动量不动、scheduler 照常前进(LR 必须与各 rank 的 global_step 保持同步)。

**阈值形式**:运行中几何均值的**无量纲倍数**,不是绝对值 —— 那次实跑里绝对尺度自身在 run 内从 0.80 漂到 0.25。作者试过 `exp(mu + k·sigma)`,回放只抓到 4 次中的 1 次(sigma 被它要检测的事件本身撑大,第一次尖峰把限值推到 525);改成几何均值的定倍数后 4/4 全抓,858 个健康步 0 误伤。

| 参数 | 默认 | 说明 |
|---|---|---|
| `training.grad_spike_skip` | `off` | `off` / `observe` / `on`。yaml 里 `on` **必须加引号**,否则 YAML 1.1 解析成 `True` 校验失败;命令行覆盖不受影响(`apply_overrides` 对标量走字符串透传) |
| `training.grad_spike_ratio` | `10.0` | 控制限 = 该倍数 × 运行几何均值,须 > 1 |
| `training.grad_spike_warmup_steps` | `500` | 前 N 个 optimizer step 不进入估计 |
| `training.adam_beta1` | `0.9` | |
| `training.adam_beta2` | `0.999` | 建议 `0.95`,漏网尖峰的恢复从 ~1000 步缩到 ~20 步 |

内部常量(不可配):`min_observations=50`、`decay=0.99`、`max_consecutive_skips=10`(连跳 10 步后强制接受一步并 re-baseline,同时打 `logger.error`)。所以守卫实际从第 ~550 个 optimizer step 起生效。

**可观测性**:开启后训练日志多出 `grad_spike_flagged`(累计判定异常)、`grad_spike_skipped`(累计真正丢弃)、`grad_norm_typical`、`grad_norm_limit`;关闭时这些 key 完全不出现,不污染老 run 的 key 集。

**上线路径**:先 `observe` 跑几千步看 `grad_spike_flagged` 只在已知崩塌点涨,再切 `on`;健康步被误伤就把 ratio 调到 15–20。守卫状态随 optimizer state dict 落盘、用 `.get()` 读,**老 checkpoint 能直接续训**,且**故意不校验** resume 时配置是否变更 —— 设计意图就是让已经炸过的 run 中途打开它。
