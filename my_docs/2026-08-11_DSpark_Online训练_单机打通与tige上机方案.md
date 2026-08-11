# DSpark online 训练:单机打通记录 + Qwen3.6-27B / tige 上机方案

日期:2026-08-11
环境:昇腾 NPU docker(8×NPU),SpecForge 分支 `feat/dspark-vocab-mapping`
前置文档:[`2026-08-10_DSpark_Online训练_NPU可行性与验证方案.md`](2026-08-10_DSpark_Online训练_NPU可行性与验证方案.md) —— 可行性分析与 S0–S7 验证阶梯

本文是上面那份方案的**执行结果**(第一部分)和**下一步计划**(第二、三部分)。

---

# 第一部分:单机已打通(Qwen3-8B)

## 结论

1. **online 全链路在昇腾上跑通了**:打过 patch 的 sglang(Ascend 后端,TP=4)→ Mooncake → producer/consumer → DSpark 训练步。实测 21 步,loss 2.9,`grad_norm` 0.15–6.8,`teacher_top1_prob` 0.79–0.88。
2. **上一份文档里排前两位的风险都已排除**:Mooncake 在昇腾环境可用(`store.so` 正常加载、put/get/remove 全通);online 抓的 `target_last_hidden_states` shape 与语义正确。
3. **唯一需要认真调的是容量参数**,不是代码。见下面的问题 ④。

## 遇到的五个问题

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| ① | 请求返回里没有 `spec_capture` | `spec_capture` 是**按请求**触发的,`--enable-spec-capture` 只装 sink | 请求体里带 `spec_capture` 字段 |
| ② | `E master_client.cpp:344] Client not available`,retry 20 次 | `mooncake_master` 没起。它默认 RPC 端口 **50051**,而约定用的是 35551 | 先起 master 并显式 `--rpc_port=35551` |
| ③ | `deployment.disaggregated.control_dir` 校验失败 | `mode` 写成了 `local_colocated`;`control_dir` 必填 | `mode: disaggregated` + 填 `control_dir` |
| ④ | 训到 21 步崩:`producer finished with 19 terminally failed prompt(s)`,同时 `could not drain 28 pending removal(s)` | **Mooncake 段被写满**。段 4 GB,单样本 ~100 MB,而 `in_flight_high_watermark` 默认 256 | 段提到 32 GB + 水位压到 16/8 |
| ⑤ | `chat_template: qwen3` / `mask_token_id` 静默不一致 | `qwen3` 不是注册模板名;`model.mask_token_id` 会**覆盖** draft config 且不报错 | 用 `qwen` 系列已注册名;删掉 `model.mask_token_id` 让它从 draft config 解析 |

**不是问题的两条日志**(排查时浪费过时间,记下来):

- `GET .../metadata?key=... http=404` + `Failed to retrieve segment descriptor` —— 传输引擎查自己尚未注册的段,紧接着就是 `Successfully created client ... after 1 attempt(s)`。**看"1 attempt"就知道没事**。
- `Global segment size is 0, skip mounting segment`(训练进程里)—— online 模式下客户端段被**强制置零**(`DISAGG_CLIENT_SEGMENT_SIZE=0`),存储段由 capture server 独占。

## 关键数字:单样本 feature 体积

这是所有容量参数的源头:

```
每样本字节 ≈ L × (aux层数 + 1) × hidden_size × 2      (bf16)
```

| 模型 | L | aux层 | hidden | 单样本 |
|---|---|---|---|---|
| Qwen3-8B | 2048 | 5 | 4096 | **~100 MB** |
| Qwen3.6-27B | 2048 | 5 | 5120 | **~126 MB** |
| Qwen3.6-27B | 4096 | 5 | 5120 | **~252 MB** |

Mooncake 段要能装下 `in_flight_high_watermark × 单样本`,而且对象是**硬 pin** 的(SpecForge 才是生命周期权威,LRU 不会帮你腾地方)。

## 拉起步骤(单机三步)

**① Mooncake**(单独终端,不能退)

```bash
mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=127.0.0.1 --rpc_port=35551 --http_metadata_server_port=35880 --metrics_port=35903
```

**② patched sglang capture server**(`MOONCAKE_*` 必须写在启动这一行,sink 在子进程读 `os.environ`,起来后再改无效)

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551 MOONCAKE_PROTOCOL=tcp MOONCAKE_GLOBAL_SEGMENT_SIZE=34359738368 MOONCAKE_LOCAL_BUFFER_SIZE=1073741824 python -m sglang.launch_server --model-path /opt/foundation_model/Qwen3-8B/ --dtype bfloat16 --trust-remote-code --skip-tokenizer-init --tp-size 4 --chunked-prefill-size -1 --disable-radix-cache --enable-spec-capture --spec-capture-method dflash --spec-capture-aux-layer-ids 1 9 17 25 33 --attention-backend ascend --context-length 4103 --mem-fraction-static 0.8 --host 127.0.0.1 --port 30000
```

不能改的四项:`--skip-tokenizer-init`(producer 直送 input_ids)、`--chunked-prefill-size -1`(采集拒绝分块 prefill,scheduler 里有硬校验)、`--context-length ≥ max_length + 7`、`--spec-capture-aux-layer-ids` 必须等于 draft config 的 `dflash_config.target_layer_ids`。

**③ 训练**

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 SPECFORGE_DEVICE=npu specforge train -c qwen3-8b.yaml "data.max_prompts=32" "training.max_steps=8" "runtime.in_flight_high_watermark=16" "runtime.in_flight_low_watermark=8" "runtime.producer_lease=4"
```

先跑 `--plan`(几秒,不起进程)把配置错误一次抓完,再真跑。判定:`prompts_failed=0`、干净退出、无 pending-remove 报错。

## 顺带确认的事实

- **入图(cuda/ACL graph)对采集没有收益,不要开。** producer 发的是 `{"temperature": 0.0, "max_new_tokens": 1}`(`server_capture.py:326`)—— 负载几乎 100% 是 prefill,而图捕获的收益在 decode。昇腾上 sglang 还会强制把 prefill 图编译降级成 eager(`server_args.py:2891`)。更要紧的是 patch 的钩子就插在 `logits_processor` 里,编译可能改变那段 buffer 的生命周期,导致**静默的数值错误**。
- `acc` 0.00–0.03、`ce_loss` ~10 是正常的:draft 随机初始化,`ln(151936) ≈ 11.9` 是全随机基线。冒烟阶段只看"跑得完",不看"训得好"。

---

# 第二部分:Qwen3.6-27B 冒烟

## draft config 不用改

`qwen3.6-27b-dspark.json` 原样使用。只有一个字段会影响 server 启动参数:

```json
"dflash_config": { "target_layer_ids": [1, 16, 31, 46, 61] }
```

→ server 必须用 `--spec-capture-aux-layer-ids 1 16 31 46 61`。**换模型时最容易漏这一步**;层号对不上,trainer 侧 `verify_capture_specs` 会因 aux 宽度不符拒绝(27B 应为 `5 × 5120 = 25600`)。

## YAML:offline → online 的改动

| 段 | 改什么 |
|---|---|
| `data` | **删** `hidden_states_path`,**换成** `train_data_path`(原始对话 JSONL) |
| `deployment` | `mode: local_colocated` → `disaggregated`,补整个 `disaggregated` 块 |
| `training` | 加 `tp_size: 1`(online consumer 必须为 1,target TP 配在 server 上) |
| `runtime` | 新增,压水位 |
| 其余 | `model` 段、`attention_backend: sdpa`、DSpark 损失权重全部**原样保留** |

完整 online YAML(`qwen3.6-27b-dspark-online.yaml`):

```yaml
model:
  target_model_path: /dpc/hot/model/y00830025/Qwen3___6-27B
  draft_model_config: /dpc/hot/w00958190/dspark/newly_specforge/qwen3.6_dspark/qwen3.6-27b-dspark.json
  target_backend: sglang
  trust_remote_code: true
  embedding_key: model.language_model.embed_tokens.weight
  torch_dtype: bfloat16
  # mask_token_id 删掉 —— draft config 里已是 248070,显式写会覆盖且不校验

data:
  train_data_path: /dpc/hot/w00958190/dspark/training_data/qwen3_6_600k_distill_4096.jsonl
  max_length: 2048            # 冒烟用 2048;正式跑改 4096,单样本体积翻倍
  chat_template: qwen3.5
  dataloader_num_workers: 4
  cache_dir: /dpc/hot/w00958190/dspark/newly_specforge/qwen3.6_dspark/online-001/cache

training:
  strategy: dspark
  num_epochs: 1               # 冒烟;正式跑见第三部分关于 num_epochs 的说明
  batch_size: 1
  accumulation_steps: 1
  learning_rate: 6.0e-4
  warmup_ratio: 0.04
  max_grad_norm: 1.0
  attention_backend: sdpa
  tp_size: 1
  num_anchors: 512            # 正式跑用 1024
  loss_decay_gamma: 4.0
  objective_chunk_blocks: 16
  dspark_ce_loss_alpha: 0.1
  dspark_l1_loss_alpha: 0.9
  dspark_confidence_head_alpha: 1.0
  grad_spike_skip: "on"       # 见前一份文档附录 A;yaml 里 on 必须加引号
  grad_spike_ratio: 10.0
  adam_beta2: 0.95
  save_interval: 500
  log_interval: 10
  dist_timeout: 30
  seed: 42

tracking:
  report_to: tensorboard

run_id: qwen3.6-27b-dspark-online-001
output_dir: /dpc/hot/w00958190/dspark/newly_specforge/qwen3.6_dspark/online-001/output

deployment:
  mode: disaggregated
  trainer:
    nnodes: 1
    nproc_per_node: 4
  disaggregated:
    control_dir: /dpc/hot/w00958190/dspark/newly_specforge/qwen3.6_dspark/online-001/control
    consumer_state_dir: /tmp/specforge/online-001/consumer-state
    backend: mooncake
    server_urls:
      - http://127.0.0.1:30000
    mooncake_metadata_server: http://127.0.0.1:35880/metadata
    mooncake_master_server_addr: 127.0.0.1:35551
    mooncake_protocol: tcp

runtime:
  producer_lease: 4
  producer_concurrency: 1
  in_flight_high_watermark: 16
  in_flight_low_watermark: 8
```

> **词表裁剪**:正式跑若用 `qwen3.6-27b-dspark-vocab-64000.json`,必须同时给 `model.vocab_mapping_path`。offline 可以从落盘 feature 现建映射,**online 没有那一遍,只能读现成文件**。

## 单机 27B 冒烟

一个节点 8 卡:server TP=4 吃 0–3(27B bf16 约 54 GB,TP=4 每卡 ~13.6 GB),trainer 4 卡吃 4–7。

server 启动在单机三步的 ② 基础上改四处:`--model-path` 换 27B、`--spec-capture-aux-layer-ids 1 16 31 46 61`、`--context-length 2055`、前面加 `GDN_ATTN_BACKEND_TRITON=1`(27B 才需要,mega_chunk_gdn C++ kernel 不支持它在 TP=4/8 下的 head 配比)。

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 SPECFORGE_DEVICE=npu specforge train -c qwen3.6-27b-dspark-online.yaml "data.max_prompts=32" "training.max_steps=8"
```

判定同 8B。**过了之后再做一次 online/offline 数值比对**(同一条 prompt 两条路各采一次,`target_last_hidden_states` 逐元素比,bf16 容差 2e-2)—— 这一步到现在还没做过。

---

# 第三部分:tige 8 节点上机

前提:**无法进容器调试**,一切靠脚本;两个 job,job1 = mooncake + sglang,job2 = 训练;地址在 job1 起来后才知道,不能手改 YAML。

## 先算三笔账

**① 存储**。L=2048 时单样本 126 MB。每个 capture server 贡献自己的 `MOONCAKE_GLOBAL_SEGMENT_SIZE`,总容量 = 段大小 × server 数。取 `in_flight_high_watermark = 64` 则需要 ~8 GB 驻留,每 server 给 32 GB 有充足余量。

**② 网络**。feature 全部经 Mooncake 跨节点搬运。trainer 世界 W、每步耗时 T 秒,则聚合带宽 ≈ `W × 126MB / T`。W=24、T=2s → **1.5 GB/s ≈ 12 Gbps**;L 改 4096 直接翻倍。**25GbE 会成为瓶颈**,集群若有 RoCE 就把 `mooncake_protocol` 改成 `rdma` 并配 `mooncake_rdma_devices`。这是 offline 完全不存在的成本项,必须先确认网络。

**③ 算力配比 —— 也是最该先想清楚的一件事。**

从 8B 实测外推:1 个 TP=4 server(4 卡)产 ~2.0 样本/s,2 个 trainer 卡消耗 ~1.5 样本/s,即**约 2:1 卡数配比**。换到 27B:capture 侧算力需求 ×3.4(27B/8B),training 侧 ×1.6(draft hidden 5120/4096、vocab 248320/151936),**配比要拉到约 4:1**。

也就是说 8 节点里大约 **6 节点做 capture、2 节点做训练**。这个结论值得停一下:

> **`num_epochs: 6` 在 online 下意味着把 27B 的 prefill 做 6 遍。** offline 是采一次、训 6 轮;online 是每轮都重采。同样的训练量,online 的 target 算力开销是 offline 的 6 倍。
>
> online 的收益是**省掉落盘**(按上面的公式,600k 样本的 hidden 是十几 TB 量级)和**数据可以无限流式**。如果目标仍是"固定 600k 数据训 6 轮",**offline 在算力上明显更划算**;online 真正的用武之地是单轮流式、数据量远大于磁盘容量的场景。
>
> 建议:上机前先明确这次是要"验证 online 可用"还是"用 online 出模型"。若是后者,把 `num_epochs` 降到 1 并相应扩大数据量,再谈配比。

**起步配比建议:capture 5 节点(10 个 TP=4 server)+ 训练 3 节点(24 rank)**,然后按下面的方法校正。

**怎么校正**:看 producer 日志的 `pool drained ... elapsed=` 和 trainer 的 `s/step`。producer 长期空转 → capture 过剩,把节点还给训练;trainer 长期等数据(`in_flight` 贴着低水位)→ capture 不足。

## 地址传递:共享目录 rendezvous

不改 YAML,靠两条已有机制:

| 要传的东西 | 机制 | 依据 |
|---|---|---|
| Mooncake 端点 | **环境变量**,优先级高于 YAML | `launch_plan.py:322` `value = base_env.get(name, configured)` |
| `server_urls` | **点分 CLI override**,值以 `[` 开头会走 `yaml.safe_load` | `schema.py:928` |
| trainer `master_addr` | 同上,用 tige 注入的 `MASTER_ADDR` | |

所以 YAML 里那几行只是占位,运行时全部被覆盖。约定一个共享根目录:

```
RUN_ROOT=/dpc/hot/w00958190/dspark/online-001
  ├── mooncake.env          # job1 node0 写
  ├── endpoints/*.url       # 每个 server 一个文件
  └── .cluster_ready        # 全部就绪的栅栏
```

**`control_dir` 必须在共享盘上** —— `nnodes > 1` 时 rank inbox 会放在 `control_dir/inboxes`(`launch_plan.py:283`);`consumer_state_dir` 必须**节点本地**(SQLite/WAL)。

## Job 1:capture 集群(5 节点)

```bash
#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=/dpc/hot/w00958190/dspark/online-001
EP_DIR="$RUN_ROOT/endpoints"; mkdir -p "$EP_DIR"
MY_IP=$(hostname -I | awk '{print $1}')
CAPTURE_NNODES=${CAPTURE_NNODES:-5}
SERVERS_PER_NODE=2                      # 每节点 2 个 TP=4 server
TARGET=/dpc/hot/model/y00830025/Qwen3___6-27B
AUX_IDS="1 16 31 46 61"                 # = draft config 的 target_layer_ids
MAX_LENGTH=2048

# --- node 0 起 mooncake,并公布端点 ---
if [[ "${NODE_RANK}" == "0" ]]; then
  mooncake_master --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 --rpc_port=35551 \
    --http_metadata_server_port=35880 --metrics_port=35903 &
  sleep 10
  cat > "$RUN_ROOT/mooncake.env" <<EOF
export MOONCAKE_METADATA_SERVER=http://${MY_IP}:35880/metadata
export MOONCAKE_MASTER_SERVER_ADDR=${MY_IP}:35551
export MOONCAKE_PROTOCOL=tcp
EOF
  touch "$RUN_ROOT/.mooncake_ready"
fi
while [[ ! -f "$RUN_ROOT/.mooncake_ready" ]]; do sleep 3; done
source "$RUN_ROOT/mooncake.env"
export MOONCAKE_LOCAL_HOSTNAME="$MY_IP"
export MOONCAKE_GLOBAL_SEGMENT_SIZE=34359738368
export MOONCAKE_LOCAL_BUFFER_SIZE=1073741824
export GDN_ATTN_BACKEND_TRITON=1

# --- 每节点起 SERVERS_PER_NODE 个 TP=4 server ---
for i in $(seq 0 $((SERVERS_PER_NODE-1))); do
  devs=$(seq $((i*4)) $((i*4+3)) | paste -sd,)
  port=$((30000+i))
  ASCEND_RT_VISIBLE_DEVICES="$devs" python -m sglang.launch_server \
    --model-path "$TARGET" --dtype bfloat16 --trust-remote-code \
    --skip-tokenizer-init --tp-size 4 \
    --chunked-prefill-size -1 --disable-radix-cache \
    --enable-spec-capture --spec-capture-method dflash \
    --spec-capture-aux-layer-ids $AUX_IDS \
    --attention-backend ascend --context-length $((MAX_LENGTH+7)) \
    --mem-fraction-static 0.8 --host 0.0.0.0 --port "$port" \
    > "$RUN_ROOT/logs/server-${NODE_RANK}-${i}.log" 2>&1 &
done

# --- 健康检查后公布 URL ---
for i in $(seq 0 $((SERVERS_PER_NODE-1))); do
  port=$((30000+i))
  until curl -sf "http://127.0.0.1:${port}/health" > /dev/null; do sleep 10; done
  echo "http://${MY_IP}:${port}" > "$EP_DIR/node${NODE_RANK}-${i}.url"
done
touch "$EP_DIR/.node${NODE_RANK}.ready"

# --- node 0 等齐所有节点,然后起 producer(CPU-only,必须是单进程) ---
if [[ "${NODE_RANK}" == "0" ]]; then
  while [[ $(ls "$EP_DIR"/.node*.ready 2>/dev/null | wc -l) -lt $CAPTURE_NNODES ]]; do sleep 5; done
  URLS=$(cat "$EP_DIR"/*.url | awk '{printf "\"%s\",", $0}' | sed 's/,$//')
  touch "$RUN_ROOT/.cluster_ready"
  specforge train -c "$RUN_ROOT/qwen3.6-27b-dspark-online.yaml" --role producer \
    "deployment.disaggregated.server_urls=[$URLS]"
else
  wait                                   # 其余节点常驻守着 server
fi
```

producer 会阻塞等 consumer 发布 dispatch quantum 握手,所以 job2 晚起没关系。

## Job 2:训练集群(3 节点)

```bash
#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=/dpc/hot/w00958190/dspark/online-001
EP_DIR="$RUN_ROOT/endpoints"

while [[ ! -f "$RUN_ROOT/.cluster_ready" ]]; do sleep 10; done
source "$RUN_ROOT/mooncake.env"
export MOONCAKE_LOCAL_HOSTNAME=$(hostname -I | awk '{print $1}')
export SPECFORGE_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

URLS=$(cat "$EP_DIR"/*.url | awk '{printf "\"%s\",", $0}' | sed 's/,$//')

specforge train -c "$RUN_ROOT/qwen3.6-27b-dspark-online.yaml" \
  --role consumer --node-rank "${NODE_RANK}" \
  "deployment.trainer.nnodes=${NNODES}" \
  "deployment.trainer.nproc_per_node=8" \
  "deployment.trainer.master_addr=${MASTER_ADDR}" \
  "deployment.trainer.master_port=${MASTER_PORT}" \
  "deployment.disaggregated.server_urls=[$URLS]" \
  "deployment.disaggregated.consumer_state_dir=/tmp/specforge/online-001/consumer-state" \
  "runtime.in_flight_high_watermark=64" \
  "runtime.in_flight_low_watermark=48"
```

`--role consumer` 下 specforge 自己 self-launch torchrun,**不要再套一层外部 torchrun**;`--role both` 会拒绝 `nnodes > 1`,多节点必须显式分 role。

## 上机前的检查单

- [ ] 两个 job 的 `RUN_ROOT` 指向同一共享路径,且**每次 attempt 全新**(`control_dir` 有残留会读到上次状态)
- [ ] `--spec-capture-aux-layer-ids` = draft config 的 `target_layer_ids`
- [ ] `--context-length ≥ data.max_length + 7`
- [ ] `MOONCAKE_GLOBAL_SEGMENT_SIZE × server 数 ≥ in_flight_high_watermark × 单样本体积 × 3`(留 3 倍余量)
- [ ] 确认集群网络带宽,决定 `mooncake_protocol` 用 tcp 还是 rdma
- [ ] 想清楚 `num_epochs` —— 见第三部分开头的算账
- [ ] **先用 2 节点(1 capture + 1 train)把这套脚本的 rendezvous 跑一遍**,再上 8 节点。脚本机制的 bug 和规模无关,小规模暴露成本低得多

## 尚未验证

| 项 | 何时有答案 |
|---|---|
| online / offline 数值一致性(代码已读实是 post-norm,但没跑过比对) | 27B 单机冒烟后 |
| 多节点 Mooncake 跨机传输(目前只验过单机 loopback) | tige 2 节点试跑 |
| 27B 长跑下的 capture/训练配比是否真是 4:1 | tige 首次全量 |
| 聚合带宽是否成为瓶颈 | tige 首次全量 |
