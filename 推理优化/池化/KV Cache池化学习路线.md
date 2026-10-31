# 00 · KV Cache 池化学习路线

> 文档范围：从"想了解池化"到"进入 L1"之前的全部内容。
> 文件路径均相对仓库根目录 `vllm-ascend`。
> 后续 Lesson 正文见 `01-Lesson1-知识点.md`。

---

## 1. 背景与目标

### 1.1 起点：此仓库里有两种"池化"

初次提出"了解池化"时，仓库内实际存在两个互不相干的特性，必须先区分：

| | **① Pooling 模型（序列池化）** | **② KV Cache 池化** |
|---|---|---|
| 池化对象 | 一条序列的 hidden states → 一个向量 | 一段前缀的 KV cache → 跨实例共享存储 |
| 目的 | 不做生成，产出 embedding / 打分 / 分类 | 降 TTFT、跨实例跨节点复用前缀 |
| 入口 | `--runner pooling` | `--kv-transfer-config` + `AscendStoreConnector` 等 |
| 官方文档 | [supported_features.md](docs/source/user_guide/support_matrix/supported_features.md#L15)、[Qwen3-Embedding.md](docs/source/tutorials/models/Qwen3-Embedding.md) | [kv_cache_pool_ascend_store.md](docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md)、[kvpp.md](docs/source/user_guide/feature_guide/kvpp.md) |
| 状态 | 🔵 Experimental | 多特性组合，按 connector 分别验证 |

**本次选择**：只学 ②，且主要看 **Mooncake** 与 **Memcache** 两种 backend。

### 1.2 KV Cache 池化的定位（背景速览）

- 定义：跨节点 KV Cache 池化存储。把 KV 写到外部存储后端（Mooncake / Memcache / Yuanrong），配合 P/D 分离，让多个 vLLM 实例共享复用，避免重复计算相同前缀。
- 关键概念：KV connector（`AscendStoreConnector`、`MooncakeConnectorV2`、`LMCacheAscendConnector`）、`kv_role`（`kv_producer` / `kv_consumer` / `kv_both`）、**bulk vs layerwise**（`use_layerwise`，`layerwise_prefetch_layers`）、`tiered_pooling_mode: "async_bypass"`、DRAM + SSD 多级池化、与 `MultiConnector` 组合。
- 代码位置：[kv_pool](vllm_ascend/distributed/kv_transfer/kv_pool)（含 `ascend_store/`、`kv_offload/`、`ucm_connector/`）。
- 排除的同名干扰项：[indexer_kpool.py](vllm_ascend/attention/indexer_kpool.py) 里的 "kpool" 是 DSA indexer 的 key pool，与池化无关。

### 1.3 学习目标

- 总体诉求：作为**新手**，尽快对 KV Cache 池化建立**比较深入**的了解，并能一步步跟着学。
- 聚焦点：Mooncake 与 Memcache。
- 逐讲目标：以路线表中每一讲"学完你要能回答"的问题作为验收标准。

---

## 2. 整体学习路线图

| 讲 | 主题 | 主战场（代码/文档） | 学完你要能回答 | 时长 |
|---|---|---|---|---|
| **L0** | 先决概念：KV cache、block_hash、池化到底在池化什么 | 本文第 4 节 | 为什么 key 天然是 hash | 40 min |
| **L1** | 全局架构：Connector / Scheduler / Worker / Backend 四层，一次请求的完整时序 | [ascend_store_connector.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L96-L121)、[pool_scheduler.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L73)、[pool_worker.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L133) | 一个 bug 该去哪个文件找 | 90 min |
| **L2** | Backend 抽象：**Mooncake = 对象存储** vs **Memcache = GVA 共享内存** | [backend/base.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py#L73-L175)、[backend/__init__.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/__init__.py#L22-L37) | 为什么两者 API 长得不一样 | 90 min |
| **L3** | Key 与命中判定：PoolKey 字段、TP/PP/group 维度、连续命中、粒度对齐 | [metadata.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L118-L213)、[pool_worker.py `lookup`](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3784-L3850) | 换个 TP 大小命中率为什么掉到 0 | 90 min |
| **L4** | 两条数据通路：Bulk（整层一次搬） vs Layerwise（逐层搬 + 预取） | [mooncake_layerwise.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_layerwise.py#L27)、[layerwise_cache_layout.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_cache_layout.py)、[attention_fence.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/attention_fence.py) | 什么时候必须用 layerwise | 120 min |
| **L5** | Mooncake 端到端部署：master、mooncake.json、SSD offload、大页、验证与调优 | [官方文档 Scenario 1](docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L65-L626) | 从零拉起一个能命中的池 | 120 min |
| **L6** | Memcache 端到端部署：MetaService、mmc-local.conf、device_sdma、SSD、分离部署 | [官方文档 Scenario 2](docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L627-L1189) + [FAQ](docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L1637-L1705) | 看一眼报错就知道卡在哪 | 120 min |
| **L7**（选） | 进阶：MultiConnector 组合 PD、KVPP、hybrid attention、TP mismatch、QoS、DCP | [kvpp.md](docs/source/user_guide/feature_guide/kvpp.md)、[coordinator.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/coordinator.py) | 能设计一个生产拓扑 | 按需 |

**推进节奏建议**：

- `L0 → L1 → L2` **必须连续学**（三天内），否则中途会断片；
- `L3 / L4` 是"读懂"的门槛；
- `L5 / L6` 是"用起来"的门槛，可跟机器解耦（先读脚本，再上机）。

---

## 3. 各 Lesson 的知识点范围、学习方式、评估标准

### L0 · 先决概念

- **知识点范围**：KV cache 是什么、为什么贵；PagedAttention 的 block / block_table / block_size（默认 128）；block_hash 与前缀复用；池化 ≠ PD 分离；Connector 的双角色与 `kv_role`；收益的量化估算；术语表。
- **学习方式**：读本文第 4 节正文 → 读第 5 节（block_hash 深入）→ 做第 4.7 / 5.6 的自检题。
- **评估标准**：能回答"为什么 key 天然是 hash"；能自己算出单 token KV 字节数与一个 block 的字节数。

### L1 · 全局架构

- **知识点范围**：四层职责（Connector 薄壳 / Scheduler 决定搬多少 / Worker 搬 / Backend 存）；一次请求的 8 个 vLLM 钩子；三条命中查询分支；命中数的三处裁剪；线程模型（load 阻塞、save 异步）；五个设计钉子。
- **学习方式**：以"锚点法"读 8 个钩子 + 四层各一个入口函数；不从头通读 `pool_worker.py`。
- **评估标准**：看到任一现象/日志，能立刻说出该去哪个文件；能回答 L1 的 5 道自检题（题目见 `01-Lesson1-知识点.md`）。原文表述："一个 bug 该去哪个文件找"。

### L2 · Backend 抽象

- **知识点范围**：`Backend` ABC 的 6 个必需项；`register_buffer` 为何必须；`put/get` 的 `list[list[int]]` 签名；layerwise 会话式协议（`batch_alloc / copy_put / commit / get_start / copy_get / get_end / lease / revoke`）；`LAYERWISE_DATA_PLANE` 的 `block_key` vs `gva`；两个入口（scheduler client vs 完整 backend）。
- **学习方式**：先读 `backend/base.py` 全文（175 行）→ 再读 `backend/__init__.py` 的注册表。
- **评估标准**：能回答"为什么两者 API 长得不一样"；能说出接入一个新后端需要实现哪些方法。

### L3 · Key 与命中判定

- **知识点范围**：`PoolKey` / `LayerPoolKey` 的字段；`block_hash` 链式结构与 `KeyMetadata` 的分工；TP / PP / DCP / group / cache_role / cache_family 各补哪个缺失维度；`infer_tp_mismatch_info` 的生效条件与 `effective_tp_size`；`cache_transfer_granularity` 的 LCM 计算；命中判定的 `hit_end` 对齐逻辑。
- **学习方式**：读 `metadata.py` 的 `KeyMetadata` / `PoolKey` / `infer_tp_mismatch_info` → 再读 `pool_worker.lookup()`。
- **评估标准**：能回答"换个 TP 大小命中率为什么掉到 0"；能判断给定场景（TP 不一致、PP、hybrid）能否命中。

### L4 · Bulk vs Layerwise

- **知识点范围**：两条数据通路的时序差异；per-layer 同步原语；`use_layerwise` 只在 Prefill 支持、强制 PIECEWISE graph；`block_key` 与 `gva` 两种数据面；attention fence 与预取距离。
- **学习方式**：读 `kv_transfer.py` 的线程类族 + `mooncake_layerwise.py` 的 `LAYERWISE_DATA_PLANE` → 结合 `attention_fence.py`。
- **评估标准**：能回答"什么时候必须用 layerwise"；能说清 layerwise 为什么只能 prefill。

### L5 · Mooncake 部署

- **知识点范围**：环境准备（`hccn.conf`、`PYTHONHASHSEED` 同步）、`mooncake.json` 与 `mooncake_master`、多租户、SSD Offload（embedded real client 模式）、大页清理与设置、warm-up、验证与调优、`ASCEND_GLOBAL_RESOURCE_CONFIG` 的 `store` 段（Store/PD 流量分离）。
- **学习方式**：有 NPU → 按文档 Step 1~5 真机跑；无 NPU → 只读脚本，并用 UT 模拟全流程（见 8.2）。
- **评估标准**：能回答"从零拉起一个能命中的池"。

### L6 · Memcache 部署

- **知识点范围**：前置检查、软件安装、`mmc-local.conf` 配置、MetaService、SSD Cache、`device_sdma` 协议、Memcache 与 vLLM 分离部署（仅 A3）、UBS IO 内存池调优、FAQ 7 类问题。
- **学习方式**：同 L5。
- **评估标准**：能回答"看一眼报错就知道卡在哪"。

### L7（选做）· 进阶

- **知识点范围**：`MultiConnector` 组合 PD；KVPP；hybrid attention / Mamba；TP mismatch；QoS（`qos_priority` 0~4）；DCP；AscendStore 已知问题。
- **评估标准**：能设计一个生产拓扑。

---

## 4. L0 正文：先决概念速通

### 4.1 三张图（后续每讲都挂在这三张图上）

**图 1：四层结构**

```
┌──────────────── vLLM 引擎进程 (EngineCore) ────────────────┐
│ Scheduler → AscendStoreConnector(scheduler role)           │
│                    → KVPoolScheduler                       │
│                        store_scheduler = Backend.create_scheduler_client()
│                        · block_hash → PoolKey              │
│                        · batch_is_exist() → 命中几个 token │
└───────────────────────────┬────────────────────────────────┘
                            │ ZMQ IPC  ipc://.../lookup_rpc_port_{port}_dp_rank{N}
┌───────────────────────────┼──── vLLM Worker 进程 (每 TP rank) ──────────────┐
│ Worker → AscendStoreConnector(worker role) → KVPoolWorker                    │
│            · register_kv_caches()  把 device KV 地址交给 backend            │
│            · start_load_kv / wait_for_layer_load                            │
│            · save_kv_layer / wait_for_save / get_finished                   │
│                        ↓                                                     │
│                  Backend: mooncake | memcache | yuanrong                     │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            ↓
             外部存储服务: Mooncake Master/Store   或   Memcache MetaService
```

**图 2：一次"命中池子"的请求**

```
请求到达
 ├(1) scheduler: request.block_hashes → PoolKey.to_string() → batch_is_exist
 │        ← 连续命中的整块数 × block_size = num_hit_tokens
 ├(2) scheduler: 命中部分标记为"外部已算"，只调度剩余 prefill token
 ├(3) worker:    start_load_kv(metadata)        发起 get
 ├(4) worker:    wait_for_layer_load()          算第 0 层前必须到位
 ├(5) forward:   attention 读本地 KV cache
 ├(6) worker:    save_kv_layer() / wait_for_save()   把新算的 KV put 回池子
 └(7) worker:    get_finished()  → 引擎释放 block、更新命中统计
```

**图 3：key/value 空间**

```
key   = "model|tp_rank|pp_rank|group|block_hash|layer|head|..."   ← 就是前缀的指纹
value = 这个 block 在 NPU 显存里的那几百 KB 原始字节
```

> **池化的本质一句话**：把"进程内的 prefix cache"外置成"跨进程、跨节点的分布式 KV 对象存储"，key 不变，只是 value 从内存搬到了别的机器。

### 4.2 KV cache 是什么，为什么贵

生成式推理里，每个 token 在每一层 attention 都要产生一对 K/V。算出前缀的 K/V 后，后续 token 直接复用，不必重算——这份缓存就是 KV cache。

- **单 token 的 KV 大小** ≈ `2 × num_layers × num_kv_heads × head_dim × dtype_bytes`（GQA 中 `num_kv_heads` 远小于 `num_heads`；MLA 存的是压缩后的 latent，形状完全不同）。
- **显存随序列线性增长，计算随序列平方增长**。所以"少算 n 个 token 的 prefill"同时省**算力**和**首 token 延迟**——这就是池化的全部经济性来源。

### 4.3 PagedAttention：block 是全套设计的基石

KV cache 被切成固定大小的 **block**（默认 `block_size=128` 个 token），由一个 **block table** 把"逻辑第 k 块"映射到"物理第 b 块"。TTFT、命中率、key 粒度，全都以 block 为单位。

> **池化只搬"整块"**。半块命中在这里是不存在的概念，后面 `cache_transfer_granularity`、`hit_end` 对齐都是在守这条规则。

### 4.4 block_hash：为什么 key 天然就是一个 hash（摘要）

vLLM 对每个 block 算一个**链式** hash：`hash(前一块的 hash + 本块 token ids [+ 额外盐])`。于是：

- 相同前缀 → 相同 block_hash（跨进程、跨机器都一致，只要模型和分词一致）；
- 不同前缀 → hash 立刻分叉，天然做到"只在公共前缀上复用"。

这就是为什么这一整个特性的 key 生成逻辑，核心只是 [metadata.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L118-L213) 里的 `PoolKey` / `LayerPoolKey` —— 它做的事就是**把 block_hash 加上"这份数据在哪个 rank、哪一组、哪一层"的坐标，拼成一个全局唯一字符串**。（完整推导见第 5 节。）

### 4.5 池化 ≠ PD 分离（新手最容易混的一对）

| | **PD 分离** | **KV Cache 池化** |
|---|---|---|
| 数据流向 | P → D，一对一 | 任意实例 → 池子 → 任意实例，多对多 |
| 生命周期 | 请求级（传完就丢） | 跨请求、跨实例，可持久化（DRAM/SSD） |
| 目的 | 让 P 和 D 独立扩缩容 | 让相同前缀只算一次，降 TTFT |
| 典型 connector | `MooncakeConnectorV1/V2` | `AscendStoreConnector` |
| 关系 | 可**组合**：`MultiConnector` 里同时挂两者 | 同左 |

### 4.6 两个必须先建立的接口概念

**(a) Connector 有"两个角色"。** 同一个 `AscendStoreConnector` 会被实例化两次：一次在调度侧（算 key、查命中），一次在每个 worker 侧（搬字节）。很多"为什么这里也有一个同名方法"的困惑都来自这一点。

**(b) `kv_role` 决定这个实例是写还是读。** 三个值：`kv_producer`（写）、`kv_consumer`（读）、`kv_both`（又写又读）。注意它和 PD 分离里的 P/D 身份**不是绑定关系**：文档里 Decode 节点也能写池子，靠 `consumer_is_to_put` / `save_decode_cache` 两个开关（[参数表](docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L1577-L1591)）。

### 4.7 算一笔账（建立数量感）

假设 7B 模型、GQA 8 个 KV head、32 层、BF16：单 token KV ≈ `2 × 32 × 8 × 128 × 2B = 128 KB`。一个 128 token 的 block ≈ **16 MB**。

- 命中 1000 个 token → 省 1000 token 的 prefill 计算 + 复用 125 MB 传输；
- 100 个并发请求共享同一段 8K 前缀 → 池子能替你省掉 99 次 × 8K token 的 prefill。

由此自然引出 L3 / L4 的核心问题：**16 MB 一个对象，整层搬还是逐层搬？搬的时候 attention 在算什么？**

### 4.8 术语表（遇到不认识的先查这里）

| 术语 | 含义 |
|---|---|
| pool | 外部 KV 存储 |
| backend | 具体存储实现 |
| connector | vLLM 接入层 |
| lookup | 查命中 |
| hit_tokens | 可跳过的 prefill token 数 |
| bulk | 整层一次搬 |
| layerwise | 逐层搬 |
| block_key | 每块一个独立 key（Mooncake 方式） |
| GVA | 全局虚拟地址（Memcache 方式） |
| producer / consumer | 写 / 读角色 |

### 4.9 L0 自检 5 题

1. `block_size=128`，某请求 1000 个 token，池子里前 6 个 block 命中、第 7 个缺失。`lookup()` 返回多少？为什么不是 768 以外的数？
2. 一个 block 的 KV 字节数怎么算？换成 MLA 模型（如 DeepSeek 系）公式哪里变了？
3. 为什么 key 里必须带 **TP rank**？如果 P 节点 TP=8、D 节点 TP=4，直接复用会出什么问题？
4. 池化能省的是 prefill 的**计算**还是 decode 的**显存**？说清楚边界。
5. 为什么"命中"必须按 block 对齐，而不能按 token 对齐？

---

## 5. L0 补充一：为什么 key 天然是一个 hash

> 触发问题：「3. block_hash：为什么 key 天然就是一个 hash 这一点我没明白」

### 5.1 先看需求：想复用，必须先回答一个等价性问题

池子要复用的对象是"某段前缀的 KV"。要复用，就必须能回答：

> 请求 B 的前 256 个 token，和请求 A 的前 256 个 token，**算出来的 KV 是不是同一份？**

三种可能的做法：

| 做法 | 问题 |
|---|---|
| 把 token ids 原文当 key | 一个 128K 的请求要存 128K 个整数当 key；每次查池子都要做 O(L) 的序列比较 |
| 用请求 ID / block ID 当 key | 这些 ID 是**进程内**的、每次重启都变，跨机器更不可能一致 → 根本没法跨实例复用 |
| **用一个定长指纹（hash）当 key** | 只要求：同样的输入 → 同样的指纹（确定性），不同输入 → 不同指纹（抗碰撞） |

第三种是唯一能同时满足"定长、可比、跨进程可复现"的方案。而 block_hash 恰好**就是**在算这个指纹，所以池化层不需要再做什么变换，直接拿现成的用。

### 5.2 关键在"链式"：为什么不能只 hash 本块 token

假设 `block_size = 128`，两条请求：

```
A:  [ 1, 2, 3 | 4, 5, 6 ]        ← 两个 block
B:  [ 7, 8, 9 | 4, 5, 6 ]        ← 第二个 block 的 token 与 A 完全相同
```

如果 hash 只覆盖本块 token：

- A 的第 2 块 → `H(4,5,6)`
- B 的第 2 块 → `H(4,5,6)` **完全相同** → 池子认为"这段我算过"，把 A 的 KV 直接给 B 用。

但这是**错的**：attention 是 causal 的，第 2 块里每个 token 的 KV 都依赖它前面**所有** token（这里是 `1,2,3` 还是 `7,8,9`）。KV 字节完全不同，拿来用输出就崩了。

所以真实的算法是**把父块的 hash 也喂进去**，构成一条链（Merkle chain / 增量指纹）：

```
h0 = H( NONE , tokens[0:128] )
h1 = H( h0   , tokens[128:256] )
h2 = H( h1   , tokens[256:384] )
```

于是每个 block 的 hash 编码的不是"这一块里有什么"，而是**"从第 0 个 token 到本块末尾的整条前缀"**。回到上面的例子：`h1` 因为 `7,8,9 ≠ 1,2,3` 而不同，于是 `h2` 必然不同 → 不会误命中。

> **代码佐证**：本仓库构造 KV event 时明确带父 hash —— [pool_worker.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3671-L3686) 里 `parent_hash = group_hashes[block_idx - 1]`，然后 `BlockStored(parent_block_hash=parent_hash, ...)`。上游生成侧是 `get_request_block_hasher(block_size, sha256)`（测试里随处可见，如 [utils.py](tests/ut/kv_offload/utils.py#L222)）。

### 5.3 "天然"的三个推论

**推论一：命中判定变成"沿链逐块比对，一断就停"。**
前缀复用的语义**恰好等于**"从 h0 开始连续存在"。不需要任何额外逻辑去判断"是不是前缀"——链式 hash 已经把这个语义编码进去了。这就是为什么 `lookup()` 的实现只是：从 block 0 开始查 `exists`，遇到第一个不存在的就停（[pool_worker.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3831-L3841)、[pool_scheduler.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L346-L359)）。

**推论二：跨进程、跨机器零协商。**
只要模型权重、分词、`block_size`、额外盐一致，任何一台机器对同一段 token 都会算出同一个 hash。所以池子不需要"注册元数据""分配 ID"，key 是**自证**的。这也是文档要求全节点同步 `PYTHONHASHSEED` 的根本原因（[doc](docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L79-L83)）——任何引入不确定性的东西都必须消除，否则同一段 token 在不同节点算出不同指纹，池子立刻失效。

**推论三：这是内容寻址存储（content-addressed storage）。**
KV 字节完全由「模型权重 + 前缀 token + 位置 + 层 + TP 分片」决定。而 hash 恰好覆盖了其中"前缀 token"这一维。所以 **key 就是 value 的内容标识**，和 git 的对象、Docker 的 layer digest 是同一个范式：`key = 指纹(内容)`，`put/get/exists` 全靠它。

### 5.4 那 hash 里还缺什么？→ 这就是 `KeyMetadata` 的全部意义

block_hash 只描述"token 前缀"，但一份 KV 字节还取决于别的维度。所以真正的池 key 是 **hash + 坐标**：

```
PoolKey.to_string()
  = "{model_name}@dcp:{dcp_rank}@head_or_tp_rank:{tp}@pp_rank:{pp}@group:{g}
     @cache_role:{kv}@cache_family:{f}@{chunk_hash}"
```

见 [metadata.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L118-L165)。逐项对应：

| 字段 | 补的是哪个"缺失维度" |
|---|---|
| `chunk_hash` | 前缀 token（链式，已自证） |
| `model_name` | 权重不同 → KV 不同 |
| `head_or_tp_rank` | TP 分片：第 3 个 rank 只存它负责的那几个 head |
| `dcp_rank` / `pp_rank` | 上下文并行、流水线切分 |
| `kv_cache_group_id` | 一个模型可能有多个 cache group（hybrid attention、sliding window…） |
| `cache_role` / `cache_family` | 同一段 token 的 KV 与 Mamba/SWA state 要分开 |
| `LayerPoolKey` 再 + `layer_id` | layerwise 时每层一个独立对象（[同文件 L167-L177](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L167-L182)） |

> `block_hash` 负责"这段 token 是不是同一段"，`KeyMetadata` 负责"这份 KV 是不是同一个物理分片"，两者相乘才是池子里唯一确定的那份字节。

### 5.5 三个常见误解

1. **"hash 会碰撞，所以不可靠"** —— sha256 的碰撞在实际规模下可忽略。工程上真正致命的是**语义漏项**：比如 LoRA、多模态 embedding、`cache_salt` 没进 hash 的 extra keys，导致"看起来同前缀、其实 KV 不同"的**误命中**。这类问题不会报错，只会算出错误的输出，是最难查的一类。
2. **"hash 是随机的"** —— 恰恰相反，它必须是**严格确定**的。哈希算法的"雪崩效应"只保证不同输入散开，确定性保证相同输入必得相同值，两者缺一不可。
3. **"命中率低是 hash 算得不好"** —— 极少是算法问题，几乎都是**坐标维度不匹配**：换 TP 大小、换 PP、换 block_size、换 salt，hash 链虽然对，但 `KeyMetadata` 拼出来的 key 变了，于是全 miss。这正是 L3 要专门讲的 `_build_tp_mismatch_keys_and_addrs`。

### 5.6 补充自检 3 题

1. `block_size=128`，请求有 1000 token。池子里 h0~h5 存在、h6 不存在。`lookup()` 返回多少 token？如果 h3 缺失但 h4~h6 都在呢？
2. 为什么 `block_hash` 里**不需要**再放 token 的绝对位置（position）？如果把 RoPE 的 base 改了（换 `rope_theta`），这个 hash 还够用吗？
3. 如果把 `block_size` 从 128 改成 64，老池子里的对象还能命中吗？为什么？（提示：`hash_block_size` vs `group_block_size` 是两个不同的粒度，见 [get_block_hashes](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L748-L756)）

---

## 6. L0 补充二：三个核心问题详解

> 触发问题：要求解答 L0 自检题的第 3、4、5 题。

三个问题其实都在问同一个东西的三个方面：**key 能否唯一标识一份不可变字节**（TP rank）、**这份字节省下了什么、省不了什么**（prefill vs decode）、**这份字节的边界画在哪**（block 对齐）。

### 6.1 为什么 key 里必须带 TP rank

**(1) 物理事实：KV 张量是按 head 切开的**

TP 的本质是把 attention 的 head 维切给不同 rank。所以：

```
逻辑上"第 0~127 个 token 的 KV" = H 个 head 的完整张量
物理上：
   rank0 的 block  → 只有 heads[0 : H/T]
   rank1 的 block  → 只有 heads[H/T : 2H/T]
   ...
```

（假设 `num_kv_heads=8`、`TP=4`，则每 rank 2 个 head。）

也就是说 **"同一个 block 的 KV"在物理上根本不是一份，而是 T 份互不重叠的字节**。

**(2) 如果 key 不带 TP rank 会发生什么**

T 个 rank 会拿**同一个 key** 去 `put` 各自不同的字节：

- 后写的覆盖先写的（对象存储语义）或冲突；
- 更糟的是**读侧**：某个 rank 读到别人写的 head 分片，形状对得上（都是 `block_size × 2 × dim`），但内容是错的 head。

后果是 attention 输出错 → 打分/生成结果异常，**但全程不报错**。这是分布式 KV 系统里最典型的一类静默错误。

所以 `KeyMetadata.head_or_tp_rank`（[metadata.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L118-L133)）的意义是：**把对象粒度定义为"某一个 rank 的那一份分片"**。池子里的对象数 = `TP × 块数 × 层数`。

**(3) P 节点 TP=8、D 节点 TP=4 直接复用会怎样**

key 空间对不上：

| | 每个对象的 head 数 | key 里的 rank 编号 |
|---|---|---|
| P (TP8) | `8/8 = 1` 个 head | 0..7 |
| D (TP4) | `8/4 = 2` 个 head | 0..3 |

D 去查 `...@head_or_tp_rank:0` 只能拿到 **1 个 head** 的字节，另一个 head 没有任何对象能提供 → **形状不匹配或直接 miss**。这里默认**不会崩**，而是退化成"各算各的 key"，表现为**命中率归零**。所以换 TP 之后收益突然消失，第一嫌疑就是这里。

**(4) 代码怎么修的：统一到 effective TP 空间**

`infer_tp_mismatch_info()`（[metadata.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L80-L113)）定义了：

```
effective_tp_size     = max(P_tp, D_tp)                  # 例：max(8,4)=8
effective_heads/rank  = num_kv_heads / effective_tp_size # 8/8 = 1
local_heads/rank      = num_kv_heads / local_tp_size     # 8/4 = 2
num_sub_keys          = local / effective                # 2/1 = 2
```

然后每个本地 rank 把自己那 2 个 head **拆成 2 个 sub-key**，key 里写的是 **effective rank**：

```python
effective_rank = self.tp_rank * self.num_sub_keys + sub_idx   # TP4 的 rank0 → 0,1
all_keys.append(self._make_sub_key_str(base_key, effective_rank))
addrs, sizes = self._build_strided_addrs(block_id, token_count, sub_idx)
```

见 [_build_tp_mismatch_keys_and_addrs](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3469-L3507)。关键在 `_build_strided_addrs`：地址不再是"整块连续"，而是**按 head 切片、带跨步**地描述同一块里的那 1 个 head。于是 TP8 和 TP4 双方看到的都是"**每个 effective rank 一个对象**"，布局一致 → 可互通。

**(5) 什么时候不可用（排障清单）**

`enabled` 要求同时满足：

```
peer_tp != local_tp
and not use_mla              # MLA 只有 1 个 latent head，无 head 可分
and not use_hybrid           # hybrid attention 不参与
and num_kv_heads >= effective_tp_size
and num_kv_heads % effective_tp_size == 0   # 必须整除
```

任一不满足 → 不报错，**静默退化为无复用的 key 空间**。所以"TP 不一致 + MLA 模型"这个组合，池化命中率必然是 0。

> **工程细节**：`_load_kv_tp_mismatch` 里有一句 `offset = self.tp_rank % len(keys)`，把 key 列表**旋转**一下再发（[L3521-L3525](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3521-L3525)）。目的是让不同 rank 从不同对象开始取，避免整机所有 rank 同一瞬间都去打同一个热点对象。

### 6.2 池化省的是 prefill 的计算，还是 decode 的显存

**答案：省的是 prefill 的计算（→ TTFT 和 prefill 吞吐），不省 decode 的显存。**

**(1) 省了什么**

| 项目 | 是否受益 | 说明 |
|---|---|---|
| 重复前缀的 prefill FLOPs | ✅ 直接省 | 命中 n 个 token 就少算 n 个 token 的前向 |
| TTFT | ✅ 直接降 | 少算 = 首 token 更早出来 |
| Prefill 节点吞吐 | ✅ 提升 | 同样算力能服务更多请求 |
| P/D 生命周期耦合 | ✅ 解耦 | P 算完写池子即可释放 device KV，不必等 D 来取，也不必与 D 同生共死 |

**(2) 没省什么（边界）**

- **Decode 的显存不省**。decode 每一步都要读**整条序列**的 KV，它必须驻留在 device 上。池子里那份是**副本**，不是"外挂内存"。所以显存占用不会因为开了池化而下降。
- **不省总流量，反而多一次搬运**：device → 池子 → device。你不是免费拿到 KV，是拿带宽换算力。
- **不省 100% 的场景**。前缀几乎不重复（每条 prompt 都独一无二）时，lookup 开销 + 写池子的带宽**全是净损失**。收益完全由 workload 的**前缀重复度**决定。

**收益最大的 workload**：多轮对话（每轮前缀 = 上一轮全量）、固定 system prompt 的 agent、RAG 的公共文档前缀、批量相似 query。
**收益最差的**：每请求独立的长文本摘要、随机 prompt 压测。

**(3) 一个重要的设计信号**

`kv_load_failure_policy` 有两个值：`recompute`（加载失败就回退重算）/ `fail`（请求失败）（[参数表](docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L1577-L1581)）。这说明在设计定位上，**池子是"可失败的加速器"，不是正确性依赖**——池子挂了应该是性能退化，而不是服务不可用。

**(4) 那"省显存"是哪条线？—— KVPP，别混**

[KVPP](docs/source/user_guide/feature_guide/kvpp.md#L5-L15)（KV pipeline parallelism）解决的是另一件事：历史 KV 原本在 TP rank 之间是**复制**的，KVPP 按层把它们分散开，从而提升有效容量（文档实测 6.25× / 5.35×）。它是**省显存**的那条线，和"跨实例复用"的 KV pool 是两个正交特性（文档也明确说二者可以叠加）。

> **KV pool**：同一份前缀，**别的地方算过了**，我搬过来用 → 省算力。
> **KVPP**：同一份 KV，**我这儿存了 T 份**，没必要 → 省显存。

### 6.3 为什么"命中"必须按 block 对齐

有三层原因，从"定义"到"物理"到"代价"。

**(1) 身份粒度：半块根本没有 key**

`block_hash` 是**整块 token 算完**才有的（链式 hash 需要 128 个 token 齐全）。半块的 token 还没到齐：

- 算不出 hash → 没有 key → 池子里无从查找；
- 就算硬造一个 key，这块的 KV 之后还会被继续写（后续 token 到来）→ 它**不是不可变对象**。

而内容寻址存储的全部正确性都建立在"**对象一旦存在就不可变**"上。允许半块，就等于允许对同一 key 反复改写，得引入版本/加锁/合并语义。

**(2) 物理/寻址粒度：block 是分配与寻址单位**

block 是 PagedAttention 的分配单位（block_table 映射逻辑块→物理块），也是传输单位。一次传输在 backend 眼里就是一组 `(addr, size)` 描述符：按页描述 → 描述符数量 = 块数；非对齐范围 → 会跨页、跨 head 切片，描述符数量成倍膨胀，DMA 效率下降。

**(3) 粒度还不止 `block_size` 一个**

```python
def infer_cache_transfer_granularity(group_block_sizes, lcm_block_size, kv_cache_group_ids):
    granularities = [lcm_block_size] + [get_group_block_size(group_block_sizes, g) for g in ids]
    return math.lcm(*granularities)     # 取最小公倍数
```

见 [metadata.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L325-L333)。因为一个模型可能有**多个 cache group**（full attention / sliding window / hybrid），每组 `block_size` 可能不同，必须取 LCM 才能让所有组**同时**对齐。这就是为什么代码里到处出现 `cache_transfer_granularity` 而不是 `block_size`，以及 lookup 里那句：

```python
if ends[index] % self.cache_transfer_granularity == 0:   # 命中点必须落在粒度边界上
```

**(4) 但"传输"可以带尾巴——重要区别**

`discard_partial_chunks`（默认 `True`）控制**尾部半块**是否参与保存/传输：

```python
num_tokens_to_save = (target_token_len // granularity * granularity)   # True：丢掉尾巴
                     if discard_partial_chunks else target_token_len    # False：连尾巴一起
```

见 [metadata.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L1207-L1216)，定位尾块用 `get_partial_block_index()`（[L759-L777](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L759-L777)）。

准确说法：

> **命中判定按块对齐**（因为 key 只能在块边界定义）；
> **传输粒度可配置**为「整块」或「整块 + 尾块」。

而 layerwise 的 partial chunk 更精细：文档说 "Partial layerwise transfers copy only the token interval covered by each segment"（[L57](docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L57)）——传输可以只覆盖段内的 token 区间。

**(5) 一致性/效率**

半块命中会让**下一次**同一前缀的请求发现"上次存的是半块，这次还是要重传尾巴"，流量被反复放大；而整块命中是幂等的、可累积的。

### 6.4 三题串起来

```
Q1  key 的唯一性 ──► 少一个维度（TP rank），就是静默错数据
Q2  value 的收益 ──► 换的是算力，不是显存；无重复前缀时是净亏
Q3  粒度的一致性 ──► key 必须在块边界存在，value 才可能是不可变对象
```

池化的全部正确性可以压缩成一句：**key 必须唯一标识一份不可变字节**。

---

## 7. 计划安排

- **形式**：讲师逐讲带学，每讲给出 3~5 个代码锚点 + 自检题；学员作答或直接说"进入下一讲"。
- **时间**：L0 40 min；L1 90 min；L2 90 min；L3 90 min；L4 120 min；L5 120 min；L6 120 min；L7 按需。
  （注：路线表标题写作"6 讲"，但表格实列 L0–L7 共 8 行，其中 L7 为选做，以表格为准。）
- **节奏**：L0→L1→L2 三天内连续完成；L3/L4 为"读懂"门槛；L5/L6 为"用起来"门槛，可与机器解耦。
- **交互约定**：每讲结束有自检题；答不上来则回补，而不是强行推进。
- **教案结构约定（用户 2026-10-31 明确要求）**：**每讲开头必须先给一节"全景"** —— 一张结构框图 + 3 句读图说明 + 5~6 条读图要点 + "图 ↔ 章节"对照表，然后再进细节。理由：新手没有全景就无法定位正在讲的是哪一块。参照 `Lesson2 Backend抽象.md` 的 §0。

## 8. 依赖的工具与资源

### 8.1 阅读资源

- 官方文档：[kv_cache_pool_ascend_store.md](docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md)（三套 backend 的部署 + 参数表 + FAQ）、[kvpp.md](docs/source/user_guide/feature_guide/kvpp.md)、[supported_features.md](docs/source/user_guide/support_matrix/supported_features.md)。
- 代码主目录：[vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/)。
- 外部参考（文档 FAQ 中给出）：Mooncake Store Deployment Guide、SSD Offload、HIXL 常见问题定位手册、Memcache FAQ。

### 8.2 可运行验证（重要）

`tests/ut/kv_transfer/` 下有一个**真调度 + 真 connector + 真线程，只 mock 掉 device 和 Mooncake 服务**的测试：

[test_ascend_store_layerwise_lifecycle.py](tests/ut/kv_transfer/test_ascend_store_layerwise_lifecycle.py#L1-L52)（其 docstring 自述："Only device operations, model computation and the external Mooncake service are simulated."）

```bash
pytest tests/ut/kv_transfer/test_ascend_store_layerwise_lifecycle.py -x -q
```

学习用法：先让它跑绿，然后改一个数字（block_size、TP、命中块数），看断言怎么炸。

同目录另有 3 个更小的 UT 可作补充：`test_layerwise_dcp_sharding.py`、`test_layerwise_multi_cache_spec.py`、`test_layerwise_pp_global_layers.py`。

e2e 参考：[test_kvpp_pool.py](tests/e2e/cases/features/kv_pool/test_kvpp_pool.py)、[test_qwen3_30b_acc.py](tests/e2e/cases/features/kv_pool/test_qwen3_30b_acc.py)、[kv_pool/config.py](tests/e2e/common/kv_pool/config.py)。

### 8.3 硬件依赖

L5/L6 真机部署需要 NPU（A2 / A3）+ Mooncake 或 Memcache 服务。**若无环境，L5/L6 改为"读脚本 + 用 mock 服务和 UT 模拟全流程"。**

---

## 9. 当时做过的关键取舍及理由

| 取舍 | 决定 | 理由 |
|---|---|---|
| 两种"池化"先学哪个 | 先学 **KV Cache 池化**（用户选择），Pooling 模型只保留对比表 | 用户明确指定第二个方向，且聚焦 Mooncake / Memcache |
| 是否先讲架构还是先讲概念 | **先概念（L0）再架构（L1）** | 新手直接看 5000 行代码会劝退；先把 KV / block / hash / 收益边界建立起来 |
| 代码怎么读 | **锚点法**：每讲讲 3~5 个入口函数，顺调用链追 | `pool_worker.py` 有 3881 行，通读是"劝退式读法" |
| 动手验证用什么 | **mock 化 UT**（CPU 可跑），改数字观察断言 | 无需 NPU 与外部服务，成本最低且能验证真实调度/线程逻辑 |
| 每讲后是否设评估 | **设自检题**，答不上来就回补 | 保证"比较深入的了解"不是错觉 |
| L5/L6 怎么讲 | 取决于是否有 NPU：读脚本 or 真机 | 避免课程被环境阻塞 |
| 三个核心问题（TP/收益/对齐）在哪讲 | 在 L0 补充里**提前讲透**，而不是留到 L3/L4 | 它们决定后面所有判断，属于"必须先建立的心智模型" |
| hash 链式结构的讲法 | 用"两条请求 A/B 的第二块 token 相同但前缀不同"作反例 | 只用一句话说明"为什么不能只 hash 本块"，最直观 |

---

## 10. 未决事项

1. **NPU 环境是否可用**（曾明确询问："你手上有可用的 NPU（A2/A3）和 Mooncake/Memcache 服务吗？"）——未答复，直接影响 L5/L6 的授课形态。
2. L0 自检 5 题、补充 3 题尚未作答；其中第 3/4/5 题已在第 6 节给出详解。
3. 是否将本课程落盘为文档——本文件即该事项的产出。