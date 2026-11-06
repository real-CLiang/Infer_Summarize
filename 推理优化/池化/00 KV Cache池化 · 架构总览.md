# 00 · KV Cache 池化 · 架构总览（vLLM-Ascend）

> 用途：**一页纸建立全局，随时回查。** 细节回到 L0–L4 原文与官方文档。
> 范围：`推理优化/池化/` 下全部笔记 + `vllm-ascend` 仓库 `kv_pool/ascend_store/`。
> 所有文件路径相对仓库根 `vllm-ascend`；行号于 2026-11-02 核对。
> 状态：**源码阅读 + 官方文档，未在 NPU 真机验证**（见 §12）。

---

## 0. 这份文档怎么用

| 你有多少时间 | 读哪里 |
|---|---|
| 3 分钟 | §1 的表 + §11 的十条判断 |
| 20 分钟 | §2 三张图 + §3/§4 的 key 与命中 + §6 的后端对照 |
| 排障 | §10 决策树 + §9 参数表 |
| 要改代码 / 接新后端 | §2.2 分层 + §6.1 契约 + §5.3 三套数据面 |

**一句话总纲**：

> 把"进程内的 prefix cache"外置成"跨进程、跨节点的分布式 KV 存储"，**key 不变，只是 value 搬到了别的机器**；省的是 prefill 的算力，换的是带宽与延迟。

---

## 1. 定位：池化在推理栈的哪一层

### 1.1 先排掉四个同名概念

| 名词 | 是什么 | 和本文的关系 |
|---|---|---|
| **KV Cache 池化**（本文） | 跨节点 KV 共享存储，降 TTFT | ✅ 主题 |
| **Pooling 模型 / 序列池化** | `--runner pooling`，把 hidden states 池化成向量做 embedding | ❌ 同名不同物，入口都不同 |
| **PagedAttention 的 block pool** | 本卡内的空闲显存块分配池 | ❌ 只是本卡分配器 |
| **KV 向量池化（算法层）** | 对一段 token 的 K/V 做 mean/max 压缩 | ❌ 有损压缩，不是资源共享 |
| **KVPP** | 把 TP rank 间**复制**的历史 KV 按层分散，省显存 | ❌ 正交特性，可叠加（见 §1.4） |

> 记忆点：**池化 = 资源共享（无损）；KVPP = 去冗余（省显存）。** 两条线互不替代。

### 1.2 与 vLLM 原生 Prefix Caching 的关系：上下两层，不是替代

```
        用户请求
           │
           ▼
  ┌──────────────────────────────┐
  │ Prefix Cache（vLLM 原生 APC） │  回答：这段前缀算过吗？
  │ 链式 hash + 本卡 block 表      │
  └──────────────┬───────────────┘
                 │ 本地 HBM 没命中
                 ▼
  ┌──────────────────────────────┐
  │ KV Cache 池（本文）            │  回答：算过的 KV 存在哪、能不能跨设备取？
  │ 跨卡/跨实例/跨节点 + 分级介质   │
  └──────────────────────────────┘
```

| | 原生 Prefix Caching | KV 池化 |
|---|---|---|
| 作用域 | 单实例、单 Worker、本卡 HBM | 跨实例、跨节点、DRAM/SSD |
| 缓存上限 | 本卡分给 KV 的显存 | 集群聚合容量 |
| 远程访问 | 不支持，跨卡即 miss | 支持（RDMA / HCCS / UB） |
| 冷热分级 | 无 | 有 |
| 定位 | **查找策略** | **存储底座** |

**官方口径**：池化必须与 Prefix Cache 同时开启——池子承载前缀缓存，不替代它。

### 1.3 与 PD 分离的关系：正交，可组合

| | PD 分离 | KV 池化 |
|---|---|---|
| 数据流向 | P → D，一对一 | 任意实例 → 池 → 任意实例，多对多 |
| 生命周期 | 请求级，传完即弃 | 跨请求/跨实例，可持久化 |
| 目的 | P/D 独立扩缩容 | 相同前缀只算一次 |
| 典型 connector | `MooncakeConnectorV1/V2` | `AscendStoreConnector` |
| 组合方式 | `MultiConnector` 同时挂两者 | 同左 |

> ⚠️ **`MooncakeConnector*`（P/D 搬运工）与 `MooncakeBackend`（池子后端）是两套东西**，名字像但职责不同。

### 1.4 收益边界：省什么、不省什么

| 项目 | 是否受益 | 说明 |
|---|---|---|
| 重复前缀的 prefill FLOPs | ✅ 直接省 | 命中 n 个 token 就少算 n 个 |
| TTFT | ✅ 直接降 | 少算 = 首 token 更早 |
| P 节点吞吐 | ✅ 提升 | 同算力服务更多请求 |
| P/D 生命周期耦合 | ✅ 解耦 | P 写完即可释放 device KV |
| **Decode 的显存** | ❌ **不省** | decode 每步要读整条序列，KV 必须驻留 device；池里那份是副本 |
| 总流量 | ❌ 反而多一次 | device → 池 → device，是拿带宽换算力 |
| 无重复前缀的负载 | ❌ 净亏 | lookup 开销 + put 带宽全是成本 |

**收益公式**：

```
收益 ≈ 命中前缀 token 数 × 每 token prefill 成本 × 命中次数
       −（put 开销 + get 开销 + 带宽竞争）
```

**设计定位**：池子是"**可失败的加速器**"，不是正确性依赖 → `kv_load_failure_policy=recompute` 是它的定位声明。

---

## 2. 总体架构

### 2.1 部署视图（PD 分离 + 池化 + 多实例）

```
                          ┌────────────────┐
       请求 ─────────────► │  负载均衡 / LB  │
                          └───────┬────────┘
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
          ┌───────────────────┐        ┌───────────────────┐
          │   P 实例 #1        │        │   P 实例 #2        │  ← 可水平扩展
          │   EngineCore       │        │   EngineCore       │     无状态
          │   + Worker×TP rank │        │   + Worker×TP rank │
          └──┬─────────────┬───┘        └──┬─────────────┬───┘
             │             │               │             │
     P2P 推KV│        put/get         P2P  │        put/get
             ▼             ▼               ▼             ▼
      ┌────────────┐   ╔═══════════════════════════════════╗
      │ D 实例 ×N   │   ║  KV 池                            ║
      │ (decode)   │   ║  Mooncake / Memcache / Yuanrong    ║
      └────────────┘   ╚═══════════════════════════════════╝
```

**三个要点**：

1. **P 层是无状态计算层**：本地 HBM 的 prefix cache 随实例上下线消失 → 池子是唯一稳定的 KV 沉淀层。
2. **P→D 与 P→池 是两条数据流**：会抢同一份 HCCS/RDMA 带宽，A3/950 上要配 Store/PD 流量分离（`ASCEND_GLOBAL_RESOURCE_CONFIG` 的 `store` 段）。
3. **默认只有 P 侧做 put/get**：D 侧通过 P2P 拿 KV（避免双份带宽）；机制上 D 也能写（见 §7.4）。

### 2.2 进程与代码分层（"一个类实例化两次"）

```
┌──────────── EngineCore 进程（纯 CPU，不碰显存）─────────────────────────┐
│  vLLM Scheduler                                                        │
│      │ get_num_new_matched_tokens / build_connector_meta                │
│      ▼                                                                 │
│  AscendStoreConnector(role=SCHEDULER)   ← 薄壳：只 assert + 同名转发     │
│      ▼                                                                 │
│  KVPoolScheduler                                                       │
│      · block_hash → PoolKey                                            │
│      · exists/batch_is_exist → 命中 token 数                            │
│      · 产出 ReqMeta（搬运清单）                                          │
│      · create_scheduler_client() → 轻量 client（能查，不能搬）           │
└───────┼────────────────────────────────────────────────────────────────┘
        │ ZMQ IPC（仅 bulk 命中查询：scheduler 问 worker rank0）
        │ ipc://.../lookup_rpc_port_{port}_dp_rank{N}
┌───────┼──────── Worker 进程 × TP rank（手里才有显存地址）────────────────┐
│       ▼                                                                │
│  AscendStoreConnector(role=WORKER) → KVPoolWorker                      │
│      · register_kv_caches() → Backend.register_buffer（pin 显存）        │
│      · start_load_kv / wait_for_layer_load                             │
│      · save_kv_layer / wait_for_save / get_finished                    │
│      · 后台线程族：Sending/Recving、Layer/KeyLayer                      │
│      ▼                                                                 │
│  Backend: MooncakeBackend | MemcacheBackend | YuanrongBackend          │
└───────┼────────────────────────────────────────────────────────────────┘
        ▼ DMA（RDMA / HCCS / UB / SDMA，不经过 CPU）
   外部服务：Mooncake Master/Store  或  Memcache MetaService/LocalService
```

**四层职责**：

| 层 | 类 | 只做这一件事 | 绝不做 |
|---|---|---|---|
| Connector | `AscendStoreConnector` | 实现 vLLM 的 `KVConnectorBase_V1`，转发 | 任何业务逻辑 |
| Scheduler | `KVPoolScheduler` | 算 key、查命中、出清单 | 搬数据 |
| Worker | `KVPoolWorker` | 注册显存、起线程、按清单搬 | 决定搬多少 |
| Backend | `Backend` ABC | `exists/put/get` + layerwise 会话协议 | 知道 vLLM 是什么 |

**为什么必须拆两个进程**：只有 worker 知道 KV 的显存地址（`register_kv_caches`），只有 scheduler 知道这一步要调度多少 token。信息不在一处 → 必须有协议（`build_connector_meta` + ZMQ）。

### 2.3 一次请求的 8 个钩子

```
请求到达
 │
 ├─① get_num_new_matched_tokens   查池子 → num_external_hit_tokens
 ├─② update_state_after_alloc      命中信息挂到 RequestTracker
 ├─③ build_connector_meta          产出 ReqMeta（block_ids/hashes/token 区间）
 │            ─── SchedulerOutput ───►
 ├─④ bind_connector_metadata       Worker 收清单
 │     register_kv_caches（启动时一次）→ Backend.register_buffer
 ├─⑤ start_load_kv                 发起 get（bulk 全量 / layerwise 开会话）
 ├─⑥ wait_for_layer_load(每层)      等该层 KV 就位（仅 layerwise）
 │        forward：attention 只算未命中的 token
 ├─⑦ save_kv_layer / wait_for_save  新算的 KV put 回池子（后台线程）
 └─⑧ get_finished                  释放 block / 统计
```

**load / save 的语义**（最容易混）：

| 操作 | 方向 | 钩子 | 底层动作 | 线程 | 关键路径 |
|---|---|---|---|---|---|
| **load** | 池 → 本机 HBM | `start_load_kv` / `wait_for_layer_load` | `get` / `get_start`+`copy_get`+`get_end` / `add_lease`+`copy`+`remove_lease` | `RecvingThread` | **是**，到齐才 forward |
| **save** | 本机 HBM → 池 | `wait_for_save` / `save_kv_layer` | `put` / `put_start`+`copy_put`+`commit` / `alloc`+`copy`+`write_finish` | `SendingThread` | **否**，主线程不等 |

三个坑：**load 不等于加载模型权重**；**save 不等于落盘**（落盘是 SSD Offload 那一层）；**`load_async=true` 不等于 load 完全异步**（仍需保证本步 KV 到齐）。

---

## 3. Key：内容寻址的身份

```
key = 指纹（链式 hash，vLLM 送） × 坐标（KeyMetadata，池化自己补）
    = {model}@dcp:{d}@head_or_tp_rank:{tp}@pp_rank:{pp}@group:{g}
       @cache_role:{kv|state}@cache_family:{c1|c2}@{chunk_hash}
```

### 3.1 指纹：链式，不是"本块内容"

```
h0 = H( NONE , tokens[0:128]  )
h1 = H( h0   , tokens[128:256] )
h2 = H( h1   , tokens[256:384] )
```

- 每个 h_k 编码的是**"从第 0 个 token 到本块末尾的整条前缀"**；
- 反例（为什么不能只 hash 本块）：`A=[1,2,3|4,5,6]`、`B=[7,8,9|4,5,6]`，若只看本块，第二块 hash 相同 → 误命中，attention 读到错 KV，**不报错**；
- 由此得到：**"前缀复用" ≡ "从 h0 起连续存在"**，不需要额外判断。

**三个推论**：

1. 命中判定 = 沿链逐块比对，一断就停；
2. 跨进程、跨机器零协商（自证），所以要求全节点同步 `PYTHONHASHSEED`；
3. 这是内容寻址存储（同 git object / Docker layer digest）。

**推论 A（工程技巧）**：`group_block_size > hash_block_size` 时不重算 hash，直接取这一组的**最后一个细粒度 hash** 当大块身份（`_LazyGroupedBlockHashList`）。

### 3.2 坐标：七个维度，各补一种"字节不同"

| 字段 | 补的维度 | 缺了会怎样 |
|---|---|---|
| `model_name` | 权重/分词不同 | 换模型仍命中 → 输出错 |
| `head_or_tp_rank` | TP 按 head 切 | 读到别人的 head，**形状对、内容错** |
| `dcp_rank` | 上下文并行切分 | 复用了别人的上下文分片 |
| `pp_rank` | 流水段切层 | A 段拿到 B 段的层 |
| `kv_cache_group_id` | 多 cache group（full/SWA/hybrid） | 不同 spec 的块互相顶替 |
| `cache_role` | KV vs Mamba/SWA state | state 被当 KV 读 |
| `cache_family` | 压缩率族 | 不同压缩布局混用 |

> **判据：切过就进 key，没切就不进。** MLA 只有 1 个 latent head，KV 在 rank 间复制 → `min(tp, num_kv_heads)=1`，只需要一份对象。

### 3.3 三套 key 格式并存

| 格式 | 长相 | 谁用 |
|---|---|---|
| 完整 `PoolKey` | `@dcp:…@head_or_tp_rank:…@pp_rank:…@group:…@cache_role:…@cache_family:…@hash` | 默认路径 / bulk |
| `LayerPoolKey` | 多一个 `@layer_id:`，无 `@pp_rank` | 历史 layerwise（每层一个对象） |
| 后端协议 key | `model@hash@rank`（block_key）/ `model@[group]@hash@rank`（gva） | Mooncake / Memcache 的 layerwise |

**定位口诀**：看到 `@dcp:` → 完整路径；看到 `@layer_id:` → 历史 layerwise；看到 `@pp_rank:` / `@pp{N}` → 后端协议路径。

### 3.4 三个 block_size，别混

| 名字 | 谁决定 | 用途 |
|---|---|---|
| `hash_block_size` | vLLM 的 hash 粒度 | 算指纹 |
| `group_block_size` | 每个 cache group 的 spec | 每组按多大一块存 |
| `cache_transfer_granularity` | **LCM**(hash, 各组) | 命中点/传输必须对齐的粒度 |

```python
granularity = lcm(hash_block_size, *[group_block_size[g] for g in ids])
```

**为什么要 LCM**：命中长度必须对**所有 group** 同时成立（一个组缺就得重算）。所以代码里到处是 `cache_transfer_granularity` 而不是 `block_size`。

**为什么要对齐**：半块**没有 key**（hash 需整块 token 才算完），也不是不可变对象；硬造 key 会导致后续块边界错位、`addrs` 指到别的物理块 → 静默错数据。

> 区分两件事：**命中判定按块对齐**；**传输在 `discard_partial_chunks=False` 时可带尾块**。

### 3.5 TP mismatch：统一坐标，不是格式转换

```python
effective_tp_size     = max(P_tp, D_tp)
enabled = (P_tp != D_tp and not use_mla and not use_hybrid
           and num_kv_heads >= effective_tp_size
           and num_kv_heads % effective_tp_size == 0)
num_sub_keys          = local_heads_per_rank // effective_heads_per_rank
effective_rank        = tp_rank * num_sub_keys + sub_idx
```

例：P(TP8) / D(TP4) / heads=8 → `effective=8`、`sub_keys=2`、D 侧 rank0 → effective rank {0,1}。

- **本质**：把所有 rank 重新编号到同一坐标系，让两侧对"一个对象是什么"达成一致（不是把 2-head 对象改写成 1-head）；
- **代价**：`_build_strided_addrs` 每个 token 一个描述符 → 44 层 × 2 张量 × 128 token ≈ **11k 描述符 / 块 / sub-key**；
- **不可用时不报错**：`enabled=False` 就静默退化成"各算各的 key"，表现为**命中率归零**。所以"TP 不一致 + MLA"必然 0 命中；
- **工程细节**：`offset = tp_rank % len(keys)` 旋转 key 列表，避免全机同时打同一个热点对象。

---

## 4. 命中判定

### 4.1 公式

```
命中 = 连续（从 h0 起） + 对齐（granularity 的倍数） + 三刀
     = min over groups（多组取最小）
```

**`lookup()` 的流程**：

```
for group in kv_cache_group_ids:
    keys = 本 rank 视角的 key
    res  = exists(展开到 全 rank × 全 PP 段)
    if use_layerwise: res = check_all_layers_exists(res, num_layers)   # 每层都在才算
    if 对齐状态组(Mamba):
        hit_end = 从后往前找最远的"命中且对齐"点
    else:
        hit_end = 从前往后找第一个断点，再回退到最近的粒度边界
    hits.append(hit_end)
return min(hits)
```

三个要点：

1. **layerwise 要"每层都在"**：对象是"一块 × 一个 head 的整层栈"，缺一层就是残缺对象；
2. **两个分支对齐方向相反**：普通组"宁可少算"，状态组"从后往前找最远"；
3. **多 group 取 min**：宁可少命中，不能有一组缺（否则 attention 读到未初始化显存）。

**为什么要展开到全 rank**：一个块要判定"可用"，必须**所有 writer 都写过**。只查自己 rank 的 key，可能"自己写过、别人没写过"却判定命中 → 加载时缺 head。

### 4.2 三条查询分支：都是问"从第 0 个 token 起连续可复用多长"

```
get_num_new_matched_tokens
 ├─ kv_consumer 且不 consumer_is_to_load → 0（只写不读）
 ├─ ① block_key layerwise（Mooncake）→ _get_block_key_layerwise_hit_tokens（scheduler 本地查）
 ├─ ② gva layerwise（Memcache）      → _get_layerwise_hit_tokens（scheduler 本地查）
 └─ ③ bulk（默认）                    → scheduler 本地查 或 LookupKeyClient ──ZMQ──► LookupKeyServer(rank0)
                                                                                     └─► pool_worker.lookup_scheduler
```

**为什么 bulk 要绕 ZMQ**：只有 worker rank0 持有"真正注册过 device buffer 的 store 句柄"；而且命中计算要与 `hbm_hit_tokens`（本地 prefix cache 已命中部分）一起算。

> 记忆点：**bulk 的命中由 worker 回答，layerwise 的命中由 scheduler 自己回答。**

### 4.3 三刀：命中数永远 ≤ 理论值

```python
if num_external_hit_tokens == request.num_tokens:
    num_external_hit_tokens -= 1              # ① 不能全命中
if use_eagle and hit_reaches_final_block:
    num_external_hit_tokens -= lcm_block_size # ② 投机解码留最后一块
if len(cacheable_group_ids) != len(original_block_size):
    num_external_hit_tokens = floor_to_granularity(...)  # ③ 压缩状态需整页重算
```

1. **不能全命中**：模型至少要真算 1 个 token（采样器要最后一个 token 的 hidden state）；
2. **EAGLE/MTP 留最后一块**：最后一块 KV 会被 draft/verify 改写；
3. **粒度对齐**：只能按整块/chunk 复用。

### 4.4 一个数字走完全流程

`prompt=1000`、粒度 128、池子有前 6 块、本地已算 128：

| 步骤 | 在干什么 | 结果 |
|---|---|---|
| ① 向下对齐 | 1000 不是 128 的倍数，尾巴 104 token 凑不成整块 | 可查长度 **896** |
| ② 问池子 | 前 7 块你有几块？→ 前 6 块有 | 池子命中 **768** |
| ③ 砍三刀 | 本例未触发 | **768** |
| ④ 减本地已算 | 128 是现成的 | 需加载 **640** |
| ⑤ 返回 | 告诉引擎"这 640 个别算了" | `(640, …)` |

> 日志里看到的命中数**永远 ≤ 理论值**；差多少，就是上面哪一步砍的。

---

## 5. 数据通路：bulk vs layerwise

### 5.1 时间线对比

```
一个请求命中 6 块（6 × 22 MB = 132 MB）

bulk（默认）
  [════════ 搬 132 MB ════════] 层0 层1 … 层43
   ↑ 这段 attention 只能干等（100% 暴露在 TTFT）

layerwise（use_layerwise=true）
  [搬层0] 层0 [搬层1] 层1 [搬层2] 层2 …
   ↑等 32 MB
               ↑ 层1 已就绪，立即返回；其余藏在计算背后
```

- 首层等待：**1.4 GB → 32 MB（1/44）**（8K 请求的量级）；
- 尾部：bulk 的 put 在 forward 结束后才开始；layerwise 的第 0 层算完就开始写；
- **换的是延迟，不是带宽效率**（反而更差：44 次小 DMA 的启动开销）。

### 5.2 三套落地方式

| | `block_key`（Mooncake） | `gva`（Memcache） | 没声明（历史） |
|---|---|---|---|
| 一个对象 | 1 block × 1 head 的**整层栈** | 同左，一段 GVA 区域 | **1 block × 1 head × 1 层** |
| key | `模型@{hash}@{head}` | `模型@[group]@{hash}@{head}` | 多一个 `@layer_id` |
| 写 | `put_start` → 逐层 `copy_put(偏移)` → `commit` | `alloc` → 逐层 `copy(GVA)` → `write_finish` | 每层直接 `put` |
| 读 | `get_start` → 逐层 `copy_get` → `get_end` | `add_lease` → 逐层 `copy(G2L)` → `remove_lease` | 每层直接 `get` |
| 线程 | `LayerSendingThread`（key-major） | `LayerSendingThread`（GVA 分支） | `KeyLayerSendingThread` |

**唯一判据**：后端模块有没有声明 `LAYERWISE_DATA_PLANE`。

```python
self.use_block_key_layerwise = self.use_layerwise and self.layerwise_data_plane == "block_key"
self.use_layerwise_transfer  = self.use_layerwise and self.layerwise_data_plane == "gva"
```

> 没声明 → 两个布尔皆假 → **悄悄走历史路径，没有任何告警**。这是"开了 layerwise 却像每层一个对象"的原因。

### 5.3 两套同步原语（缺一个就是偶发精度问题）

| 原语 | 管什么 | 用在哪 |
|---|---|---|
| `threading.Event`（per layer） | 传输线程"这层搬完了"（CPU 侧） | `wait_for_layer_load` 里等 |
| `torch.npu.Event`（per layer） | **KV 字节真的写进显存了**（device 侧） | `save_kv_layer` 先 `record()`，发送线程再 `synchronize()` |

**为什么两套都要**：`save_kv_layer` 是 Python 侧被调用的，此时 NPU 上的 attention kernel 可能还没执行完。DMA 必须等 device event。

### 5.4 预取与 attention fence

```
for layer in 0..43:
    wait_for_layer_load()   # ① 提交窗口内任务 ② 等本层 Event（10s 超时只打日志）
    attention(layer)        # ③ 计算
    save_kv_layer()         # ④ record npu event → 丢给发送线程
```

- **预取窗口**：`num_prefetch_layers` 默认 `min(shared_buffers, 8)`，可用 `layerwise_prefetch_layers` 调；
- **attention fence**：attention 提交前 record 一个 NPU event，预取 DMA 等它完成才提交——因为 **DMA 与 attention 抢同一份 HBM 带宽**，无脑抢先发会拖慢本层计算；
- **10 秒超时不是失败**：只打一行日志继续等，真失败靠 `raise_if_failed()`。

### 5.5 约束与代价

| 约束 | 为什么 |
|---|---|
| **只能 Prefill** | decode 每步只算 1 token，44 次小 DMA 的启动开销 > 收益 |
| **必须 PIECEWISE 图** | FULL graph 中间没有 Python 回调点，无法在层间插同步 |
| **与 `save_decode_cache` 互斥** | 该模式要求非 layerwise（配 `consumer_is_to_put` 可绕过部分限制） |
| 小传输更多 | 44 次 vs 1 次，描述符数量与启动开销 |
| 抢带宽 | 见 attention fence |
| 峰值缓冲 | 只需几层环形缓冲，而 bulk 要装下整包 |

**什么时候"必须"用 layerwise**：① 整包等待扛不住（超长前缀 + 带宽有限）；② 要尽早发布（P 边算边写，D 更早取到前几层）；③ 显存放不下整包缓冲；④ 后端只有地址平面（`gva` 的租约/逐层可见依赖 layerwise）。

---

## 6. 存储后端

### 6.1 契约：6 个必实现 + 3 个默认 + 一组可选协议

| 档 | 内容 |
|---|---|
| **必实现 6** | `__init__`、`set_device`、`register_buffer`、`exists`、`put`、`get` |
| 默认实现 3 | `create_scheduler_client`、`close`、`batch_is_exist` |
| **会话组**（对象语义） | `put_start` / `copy_put` / `commit` / `revoke` / `get_start` / `copy_get` / `get_end` |
| **空间组**（地址语义） | `alloc` / `add_lease` / `remove_lease` / `write_finish` |
| 能力查询 | `validate_layerwise_support`、`batch_is_readable`、`get_key_info` |
| 显式泄漏 | `requires_exists_before_put` |

> 两组可选方法**几乎不重叠**：会话组属对象语义（空间在仓库手里），空间组属地址语义（空间在你手里）。**这就是"两家 API 长得不一样"的答案——不是风格，是物理模型不同。**

**三个物理约束**：

1. **`register_buffer` 是硬约束**：跨机搬运走 DMA（RDMA/HCCS/UB/SDMA），硬件只肯访问登记过（pin 住）的内存。**没登记 = 一次都搬不了**；
2. **`put/get` 的 addrs 是二维的**：外层是 key，内层是"这个 key 散成几段"。一个 44 层 × K/V 的 block ≈ 88 段，互不相邻。**一个对象 ≠ 一段连续内存**；
3. **`exists` 只返回 0/1**：其他值视为后端报错。

### 6.2 两个入口：同一层，两套权限

```
                 ┌──────────────────────────────┐
  Scheduler ───► │ create_scheduler_client()    │ 轻量：能查，不能搬
                 ├──────────────────────────────┤
  Worker    ───► │ Backend(parallel_config)     │ 完整：register_buffer + put/get
                 └──────────────────────────────┘
```

**为什么前台要轻**：① 不虚增容量（别把本机内存/SSD 捐给池子）；② 不建 NPU 上下文（否则白占一张卡）；③ 不参加 DP 同步（否则卡在等不到的信号上）。

> 这解释了 §4.2：**"能查"和"能搬"是两套权限**，所以 bulk 命中要绕 ZMQ 问 worker。

### 6.3 两种存储语义（"两家 API 长得不一样"的答案）

| 阶段 | 对象仓库（Mooncake） | 共享内存（Memcache / GVA） |
|---|---|---|
| 拿空间 | `put_start(keys, sizes)`：只登记意图，**不给地址** | `alloc(keys, sizes)`：**给地址** |
| 写 | `copy_put(keys, buffers, sizes, **对象内偏移**)` | `copy(地址数组, 本地地址, 大小, 方向)`，key 不参与 |
| 发布 | `commit(keys)` → 此刻才可见 | `write_finish(keys)` → 此刻才可读 |
| 放弃 | `revoke(keys)` | 无对称操作（未发布的会被淘汰） |
| 读前 | `get_start(keys)` | `add_lease(keys)`（租约） |
| 读后 | `get_end(keys)` | `remove_lease(keys)` |
| 能否读 | `exists`（没 commit 本就看不见） | `batch_is_readable` / `get_key_info`（需有大小且地址非 0） |

> 一句话：**对象仓库把地址藏起来，所以接口以 key 为中心；共享内存把地址交给你，所以接口以地址为中心。**

### 6.4 三个后端对照

| | **Mooncake** | **Memcache** | **Yuanrong** |
|---|---|---|---|
| 出身 | Moonshot AI 开源，Kimi 底座 | 华为 MemFabric 之上的池化共享内存 | openEuler 开源数据系统 |
| 抽象 | 对象（key → blob） | 地址（GVA + 偏移） | key + 地址 |
| 数据面 | `block_key` | `gva` | **未声明**（走历史路径） |
| 组件 | TE + Store(master) + Conductor | MetaService + LocalService | Coordinator / etcd |
| 分层 | DRAM + **SSD Offload** | HBM → 二级 DRAM → 三级 DRAM → SSD | 视部署 |
| 额外能力 | 租户配额、Store/PD 流量分离、metrics | A3 HCCS/UB、`device_sdma`、分离部署、DP init barrier | Remote H2D、FabricMem |
| `requires_exists_before_put` | True | True | **False** |
| 在仓库的角色 | 既能 P/D 搬运工，也能做池子 | **只做池子** | 只做池子 |
| layerwise 门槛 | ≥ 0.3.14 | `memcache_hybrid` ≥ 1.3.0 | — |

**Memcache 特有的坑**：`exists=1` 但读不到——数据在 SSD，没有可 DMA 的地址。命中查询用 `batch_is_readable`，真读前先 **rewarm**（临时申请 NPU 内存 → 读回 DRAM → 释放）。

### 6.5 选型

| 你的约束 | 指向 | 依据 |
|---|---|---|
| 要容量（KV 装不下 DRAM） | Mooncake | 成熟的 SSD Offload 与淘汰策略 |
| 要多租户隔离/配额 | Mooncake | `tenant_id` + 配额快照 + metrics |
| A3 超节点、追求 P 侧延迟 | Memcache | MemFabric + UB / `device_sdma` |
| 不想多引一套依赖 | Mooncake | 默认后端，生态最广 |
| P/D P2P 与池化同一套栈 | Mooncake | P/D connector 本身就是 Mooncake 的 |
| A3 上 MemCache 与 vLLM 分离部署 | Memcache | 只有 A3 支持该形态 |

**两处常见误判**：

1. ❌"逐层搬运只能选 Memcache"——Mooncake ≥ 0.3.14 也支持 layerwise，差别只有协议与版本门槛；
2. ❌"低延迟是 Memcache 独占"——Mooncake 在 Ascend 上也走 `AscendDirectTransport`（HCCS/RDMA）。

> **数据平面（怎么搬）与系统能力（装多少/多快/谁管）是两个正交层次。** 选型靠能力清单，不靠操作维度。

---

## 7. 部署形态

### 7.1 实例 / 会话 / DP rank

| 概念 | 是什么 | 影响池化哪一项 |
|---|---|---|
| **实例** | 一次 `vllm serve` 起的、独占一组 NPU 卡的副本；有自己的模型权重副本 + KV 显存 + prefix cache | **跨实例复用能否发生** |
| **会话/并发** | 同时在线的请求流 | 命中次数、带宽竞争 |
| **DP rank** | 同一实例内 `--data-parallel-size N` 起的 N 个独立 engine core（各有 scheduler 与 block manager） | 从池化视角**等价于 N 个实例** |

**"跨实例"实际是"跨 KV cache 所有者"**，四类：多副本、同实例内多 DP rank、同 rank 重启前后、本地 prefix cache 被 evict 之后。

两个反直觉点：

- **单实例 ≠ 不用池化**：前缀长、并发高、本地装不下时，池子就是二级缓存；
- **会话数多 ≠ 池化收益大**：会话粘在同一副本时，上一轮 KV 就在本地 HBM，走的是本地 prefix cache。

### 7.2 PD 分离 vs PD 混合

| | PD 分离 | PD 混合（colocated） |
|---|---|---|
| P/D 关系 | 独立进程/节点，P2P 传输 | 同一实例 |
| 池化收益 | 跨实例复用 + P 无状态化 | 跨请求复用，**无双份带宽问题** |
| 代价 | P→D 与 P→池 两条流抢带宽 | 容量与算力耦合 |

### 7.3 PD 分离下池化是否"作用更大"

**PD 放大的是"命中次数"里的实例数因子，而且是间接的**：

1. **P 层无状态化**（最硬的理由）：P 可扩缩容 → 本地 HBM prefix cache 随实例消失 → 池子是唯一稳定沉淀层；
2. **P/D 解耦**：P2P 是"算完推给某个 D"，池子是"commit 后任何 D 随时取"，支持跨时间复用；
3. **长上下文/大模型**：单 token prefill 成本高，命中绝对收益更大。

**但"更大"不等于"无代价"**：

| 代价 | 说明 |
|---|---|
| 双份带宽 | P 本来就要 P2P 推给 D，再 put 进池子是第二条流 |
| P 侧 TTFT 更敏感 | put 异步，但 **get 同步**；命中率不高时固定开销直接进 TTFT |
| 前缀重复率 | 一次性语料/生成型负载命中率≈0，纯亏 |

**真吃到红利**：前缀重复率高（系统 prompt / few-shot / 多轮）、P 实例数 > 1 且请求被摊开、上下文长（万级 token 起）、有容量兜底、带宽有余量。
**别开**：单 P 实例且本地装得下、前缀几乎不重复、短 prompt、带宽已吃紧。

### 7.4 `kv_role` 与读写开关

| 开关 | 作用 |
|---|---|
| `kv_role` | `kv_producer`（写）/ `kv_consumer`（读）/ `kv_both` |
| `consumer_is_to_put` | 让 consumer 也写（默认 False） |
| `consumer_is_to_load` | 让 consumer 也读（默认 False） |
| `save_decode_cache` | Decode 侧保存增量 KV（默认 False，与 layerwise 互斥） |

> **不是机制限制，是配置取舍**：默认只在 P 侧 put/get，是因为 D 侧每步只算 1 token，收益低且要抢带宽。

---

## 8. 运行时：线程与失败

### 8.1 线程模型

| 线程类 | 用在哪 | 语义 |
|---|---|---|
| `KVCacheStoreSendingThread` | bulk save | 异步 put，主线程不等 |
| `KVCacheStoreRecvingThread` | bulk load（`load_async=true`） | 异步 get |
| `KVCacheStoreLayerSendingThread` | `gva` layerwise save | 逐层 put |
| `KVCacheStoreKeyLayerSendingThread` | 历史路径 layerwise save | 每层一个对象 |
| `LayerRecvingThread` / `KeyLayerRecvingThread` | 对应 load | 逐层 get |

### 8.2 失败矩阵：越靠近决策越严，越靠近动作越松

| 位置 | 策略 | 理由 |
|---|---|---|
| 命中查询（scheduler `exists`） | **fail-closed**：报错停下 | 宁可不服务，也不假装 miss（否则算错/漏算） |
| 保存前过滤（`_get_missing_indices`） | **fail-open**：当"没有"直接写 | 多写浪费带宽，少写会丢块 |
| worker 侧 `lookup()` | `try/except → 0` | 只负责"我这儿能给出多少"，决策仍在 scheduler |
| load 失败 | `kv_load_failure_policy`：`recompute`（回退重算）/ `fail`（请求失败） | 池子是加速器，不是正确性依赖 |
| save 失败 | 忽略 | 只影响"下次不命中" |

### 8.3 一致性契约

- **可见性是原子的**：未 `commit` / 未 `write_finish` 的对象谁都读不到；
- **不要覆盖已提交对象**：block-key 后端会拒绝（`MMC_UNMATCHED_KEY`），甚至破坏正在被读的对象 → 所以写之前先 `exists` 过滤（`requires_exists_before_put`）；
- **读要租约**：`add_lease` 防并发回收；
- **会话要记账**：`LayerwiseSessionTracker` 记引用计数，最后一个读者退出才 `get_end`。

---

## 9. 参数速查

| 参数 | 作用 | 备注 |
|---|---|---|
| `backend` | `mooncake`（默认）/ `memcache` / `yuanrong` | 同时决定数据面（`block_key` / `gva`） |
| `use_layerwise` | 开逐层搬运 | 仅 Prefill；强制 PIECEWISE |
| `layerwise_prefetch_layers` | 预取距离 | 默认 `min(shared_buffers, 8)` |
| `load_async` | 异步 get | 不等于完全异步 |
| `kv_load_failure_policy` | `recompute` / `fail` | 建议用 `recompute`（池子可失败） |
| `save_decode_cache` | Decode 侧保存增量 | 与 layerwise 互斥 |
| `consumer_is_to_put` / `consumer_is_to_load` | consumer 侧读写开关 | 默认 False |
| `discard_partial_chunks` | 尾块是否参与传输 | 默认 True |
| `layerwise_num_shared_buffers` | 逐层环形缓冲数 | 影响预取窗口上限 |
| `memcache_dp_init_barrier` | DP 初始化同步 | 默认 True |
| `qos_priority` | 传输 QoS | 0~4 |
| `tenant_id` | Mooncake 租户命名空间 | 配额/隔离 |
| `cache_salt` | 请求级隔离盐 | 进了 hash 才不会串 |
| `enable_prefix_caching` | 原生前缀缓存 | 池化需同时开启 |

> 默认值与取值范围以官方参数表为准（`docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md`）。

---

## 10. 排障决策树

```
现象：命中率低 / 为 0
 ├─ 换过 TP / PP / block_size / backend / salt？
 │     └─ 是 → 换了 key 空间或指纹链 → 老池子必然全 miss（§3.3/§3.4/§3.5）
 ├─ TP 不一致且是 MLA / hybrid / heads 不整除？
 │     └─ 是 → tp_mismatch 静默禁用 → 命中 0（§3.5）
 ├─ 是否 kv_consumer 且没开 consumer_is_to_load？
 │     └─ 是 → 直接返回 0（§4.2）
 ├─ 命中数比理论值小？
 │     └─ 是 → 三刀（不能全命中 / EAGLE 留块 / 粒度对齐）+ 多组 min（§4.3）
 └─ 以上都不是 → 看是哪条查询分支（bulk 走 ZMQ / layerwise 本地查）

现象：结果异常但不报错
 ├─ key 少维度？（TP/PP/DCP/group/role/family/layer）→ §3.2
 ├─ buffer 没注册？（register_kv_caches 没执行）→ §6.1
 └─ 没 commit / write_finish 就有人读？→ §8.3

现象：开了 layerwise 却像"每层一个对象"
 └─ 后端没声明 LAYERWISE_DATA_PLANE → 静默走历史路径 → 看实际创建的线程类（§5.2）

现象：graph 捕获失败
 └─ 没落到 PIECEWISE → requires_piecewise_for_cudagraph 返回 use_layerwise（§5.5）

现象：exists=1 但 get 失败（Memcache）
 └─ 数据在 SSD，无可读地址 → batch_is_readable / rewarm（§6.4）

现象：传得快但 TTFT 没改善
 └─ 预取窗口太大 / 抢带宽 → layerwise_prefetch_layers、attention_fence（§5.4）

现象：池子故障导致请求失败
 └─ kv_load_failure_policy=fail → 改 recompute（§1.4）
```

---

## 11. 架构师视角的十条判断

1. **池子是可失败的加速器**，不是正确性依赖——这条决定了 failure policy、读严写松、压测口径。
2. **key 是唯一契约**：跨版本、跨拓扑、跨后端的兼容面全在 key 上。改一个字符 = 老池子零命中，所以要给 key 格式加快照测试。
3. **key = 指纹 × 坐标**。hash 管"是不是同一段 token"，坐标管"是不是同一份字节"。**少一个维度就是静默错数据**，而且不报错。
4. **命中永远 ≤ 理论值**：不能全命中、EAGLE 留块、粒度对齐、多组取 min。
5. **抽象只能把差异显式化，不能消灭它**：`requires_exists_before_put`、`batch_is_exist`、`batch_copy` 全是"底层语义不一致"的泄漏点。
6. **传输是二维的**：对象 ≠ 连续内存。这一条决定了 `addrs` 签名、TP mismatch 的 11k 描述符、MLA 的分段。
7. **layerwise 换延迟不换带宽**：44 次小 DMA 效率更差，收益全在"隐藏等待"。所以它只在"传输时间相对计算时间不可忽略"时划算。
8. **换后端 / 换 TP / 换 block_size = 换 key 空间**：这是"命中率突然归零"的头号原因。
9. **默认 P 侧 put/get，D 侧走 P2P**：避免双份带宽；要破例就得接受带宽竞争。
10. **先算收益，再谈优化**：`前缀重复率 × 上下文长度 × 实例数` 三个因子不成立时，池化是净亏。

---

## 12. 边界与未验证点

**已经确立的边界**（别越界理解）：

- 池化**不省显存**（decode 的 KV 必须驻留 device）；省显存的是 KVPP；
- 半块**没有 key**；能带尾块的是"传输"，不是"命中"；
- 池化**不替代** Prefix Caching，而是它的存储底座；
- 换后端 = 换 key 空间，两个池子的数据互不认识。

**未验证**（均来自源码阅读 + 官方文档）：

1. 全部结论**未在 NPU 真机验证**；
2. `shared_buffers` 的实际取值路径未追（预取窗口默认值只追到 `_DEFAULT_MAX_PREFETCH_LAYERS = 8`）；
3. 历史路径 `put_step` 跨步的动机（`num_kv_heads < tp_size` 时 KV 复制）是**推断**；
4. layerwise 对 TTFT 的收益只有量级估算，未实测；
5. `LayerPoolKey` 省略 `pp_rank` 是"`layer_id` 已隐含 stage"——推断，未找到设计文档；
6. worker 侧 `lookup()` 的 `try/except → 0` 与 scheduler 侧"报错"的不一致是否为有意分层，待与作者确认；
7. `Yuanrong + use_layerwise` 的行为（未声明数据面 → 历史路径）是代码推断；
8. 未运行 `pytest tests/ut/distributed/ascend_store/` 与 `tests/ut/kv_transfer/test_ascend_store_layerwise_lifecycle.py`。

---

## 13. 文档地图

| 文件 | 内容 | 什么时候读 |
|---|---|---|
| `KV Cache池化学习路线.md` | L0 先决概念、hash 链式推导、TP/收益/对齐三问 | 入门第一遍 |
| `Lesson1 全局架构学习.md` | 四层 + 8 钩子 + 三刀 + 线程 + 薄壳 | 建立骨架 |
| `Lesson2 Backend抽象.md` | 6+3+14、二维地址、两种存储语义、三套数据面 | 读后端代码前 |
| `Lesson3 Key与命中判定.md` | 七维坐标、三个粒度、TP mismatch、lookup | 排障 / 改 key |
| `Lesson4 Bulk与Layerwise.md` | 两条通路、同步原语、预取、fence、后端对照 | 性能调优 |
| `L0-L3 整体回顾.md` | 六条主线 + 八钩子×四层表 + 症状→文件 | 快速回忆 |
| `KV Cache 池化 与 vLLM 原生 Prefix Caching 对比与关联.md` | 分层关系、易混概念 | 对外解释时 |
| **本文** | 全局架构 + 决策树 + 架构判断 | 随时回查 |

**下一步建议**：L5（Mooncake 部署）/ L6（Memcache 部署）/ L7（MultiConnector、KVPP、QoS、DCP）。
