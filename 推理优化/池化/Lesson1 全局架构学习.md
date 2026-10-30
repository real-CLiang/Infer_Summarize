# 01 · Lesson 1 知识点：全局架构

> 范围：从"进入 L1"开始的全部内容，含 L1 正文与随后的 4 组答疑。
> 文件路径均相对仓库根目录 `vllm-ascend`。先修内容见 `00-学习路线.md`。

---

## 1. 学习目标

学完这一讲，看到任一条日志或任一个现象，都应该能立刻说出**该去哪个文件找**。

具体验收：

1. 能说出四层的职责边界，以及为什么 Connector 必须是"薄壳"；
2. 能按顺序列出一次请求经过的 8 个 vLLM 钩子，并指出每个钩子落在哪个文件；
3. 能说清命中数是怎么算出来的、会被哪三处裁剪；
4. 能区分 load 与 save 的同步语义，以及为什么这样设计；
5. 能认出 `Backend` 层的抽象契约与它的"泄漏点"。

---

## 2. 核心概念与原理

### 2.1 一句话架构

> `AscendStoreConnector` 是一个**薄壳**：**Scheduler** 决定"要不要搬、能搬多少"，**Worker** 决定"怎么搬、搬到哪"，**Backend** 决定"存到哪台机器、用什么语义存"。

### 2.2 角色分裂：同一个类，实例化两次

vLLM 为每个连接器实例化**两个对象**，靠 `role` 参数区分：

```python
if role == KVConnectorRole.SCHEDULER:
    self.connector_scheduler = KVPoolScheduler(...)          # 在 EngineCore 进程
else:
    self.connector_worker = KVPoolWorker(...)                # 每个 TP rank 一个进程
    if not self.use_layerwise and vllm_config.parallel_config.rank == 0:
        self.lookup_server = LookupKeyServer(self.connector_worker, vllm_config)
```

见 [ascend_store_connector.py L153-L169](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L153-L169)。

**为什么必须拆成两个进程？**

| | Scheduler 侧 | Worker 侧 |
|---|---|---|
| 在哪 | EngineCore（CPU 进程） | 每个 NPU/TP rank 一个进程 |
| 手上有 | request、block_hashes、调度预算 | **device 上的 KV 张量地址**、已注册的 store 句柄 |
| 干的活 | 算命中、决定少算多少 prefill | 真的 DMA 搬字节 |
| 不能干 | 碰显存、做几千次 exists 查询 | 改调度决策 |

分工的硬约束：**只有 worker 进程知道 KV 在显存里的地址**（`register_kv_caches`），而**只有 scheduler 知道自己这一步要调度多少 token**。两边信息不在一处，所以中间必须有协议——这就是 `build_connector_meta` 和 ZMQ 存在的理由。

### 2.3 四层职责表

| 层 | 类 / 文件 | 只做这一件事 | 绝不做 |
|---|---|---|---|
| **Connector** | [`AscendStoreConnector`](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L96) | 实现 vLLM 的 `KVConnectorBase_V1` 接口，**全部方法都是 `assert` + 转发** | 任何业务逻辑 |
| **Scheduler** | [`KVPoolScheduler`](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L73) | 生成 key、查命中、产出 `ReqMeta`（搬运清单） | 搬数据 |
| **Worker** | [`KVPoolWorker`](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L133) | 注册显存、起线程、按清单搬、报完成 | 决定搬多少 |
| **Backend** | [`Backend`](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py#L73) | `exists / put / get`，以及 layerwise 的会话式协议 | 知道 vLLM 是什么 |
| **外部服务** | Mooncake Master / Memcache MetaService | 存对象、管容量、SSD 换入换出 | 知道 KV 是什么 |

验证方式：`AscendStoreConnector` 的每个下游方法都是三行。

```python
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    assert self.connector_scheduler is not None
    return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

def build_connector_meta(self, scheduler_output):
    assert self.connector_scheduler is not None
    return self.connector_scheduler.build_connector_meta(scheduler_output)
```

[L235-L248](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L235-L248)

### 2.4 一次请求的 8 步

| # | vLLM 调用的钩子 | 我们的实现 | 发生了什么 |
|---|---|---|---|
| 1 | `get_num_new_matched_tokens` | [pool_scheduler:595](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L595-L718) | 查池子，返回**还能从池子加载多少 token**（`need_to_allocate`） |
| 2 | `update_state_after_alloc` | [pool_scheduler:720](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L720) | 引擎给请求分了新 block，把命中信息挂到 `RequestTracker` |
| 3 | `build_connector_meta` | [pool_scheduler:1002](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1002) | 生成 `AscendConnectorMetadata`（内含 `ReqMeta`：block_ids、block_hashes、token 区间）→ 随 `SchedulerOutput` 下发 |
| 4 | `bind_connector_metadata` + `register_kv_caches` | [connector:335](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L335-L345) / [worker:1203](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1203) | worker 收到清单；**启动时一次性**把 KV 显存地址交给 backend（`register_buffer`） |
| 5 | `start_load_kv` | [connector:347](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L347-L366) | 发起 `get`：bulk 全量 / layerwise 开会话 |
| 6 | `wait_for_layer_load(layer_name)` | [connector:368](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L368-L383) | **每层 attention 前**等到该层 KV 就位 |
| 7 | `save_kv_layer` / `wait_for_save` | [connector:412](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L412-L433) | 把本步新算的 KV `put` 回池子（后台线程） |
| 8 | `get_finished` | [connector:435](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L435-L445) | 告知引擎哪些请求 load/save 完成 → 释放 block、统计命中 |

**骨架就这 8 个钩子**；剩下 5000 行全在处理这 8 步里的边界情况（PP、TP mismatch、hybrid、MTP、mamba、SSD、重试）。

### 2.5 命中查询的三条分支

`get_num_new_matched_tokens` 是唯一决定"能省多少"的地方，有**三条分支**（[L612-L647](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L612-L647)）：

```
                         ┌─ kv_consumer 且不 consumer_is_to_load → 直接 0（只写不读）
                         │
 get_num_new_matched_tokens
                         │
     ┌───────────────────┼──────────────────────┐
     ▼                   ▼                      ▼
 ①block_key layerwise  ②gva layerwise        ③bulk（默认）
 (Mooncake+use_layerwise)(Memcache+use_layerwise)
     │                   │                      │
 _get_block_key_      _get_layerwise_      ┌────┴─────────────┐
 layerwise_hit_tokens hit_tokens           │ use_layerwise?   │
     │                   │                 ▼                  ▼
     └───────────────────┴────► scheduler 本地查        LookupKeyClient
                                  (store_scheduler,       ↓ ZMQ IPC
                                   create_scheduler_client)  LookupKeyServer(rank0)
                                                              ↓
                                                        pool_worker.lookup_scheduler
```

**为什么普通 bulk 要绕一圈 ZMQ？** 因为只有 worker rank0 持有"真正注册过 device buffer 的 store 句柄"，而且命中计算还要和 `hbm_hit_tokens`（本地 prefix cache 已命中的部分）一起算。server 端是一个 REP socket 循环：

```python
result = self.pool_worker.lookup_scheduler(token_len, hashes_str, kv_group_ids,
                                          use_layerwise=False, hbm_hit_tokens=hbm_hit_tokens)
response = result.to_bytes(4, "big")
```

见 [`LookupKeyServer` L513-L584](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L513-L584)。socket 路径由 `lookup_rpc_port` 决定，形如 `ipc://.../lookup_rpc_port_{port}_dp_rank{N}`（[L1246-L1261](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1246-L1261)）——**IPC 说明是同一台机器上的两个进程**，不是网络 RPC。

> **记忆点**：bulk 的命中由 worker 回答，layerwise 的命中由 scheduler 自己回答。这就是为什么 `use_layerwise` 一改，日志里 `kvpool hit tokens` 的来源都变了。

### 2.6 命中数的三处裁剪

`get_num_new_matched_tokens` 拿到命中数后**并不直接返回**，还要过三道闸：

```python
if num_external_hit_tokens == request.num_tokens:
    num_external_hit_tokens -= 1        # ① 不能全命中
if self.use_eagle and hit_reaches_final_block:
    num_external_hit_tokens -= self.lcm_block_size   # ② MTP 要把最后一块留给本地重算
if len(cacheable_group_ids) != len(original_block_size):
    num_external_hit_tokens = floor_to_granularity(...)  # ③ 压缩状态需整页重算
```

[L674-L679](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L674-L679)

1. **不能全命中**：模型**至少要真算 1 个 token**。生成模型要最后一个 token 的 hidden state → logits → 采样；embedding 模型要至少一个 token 的 hidden state 才能 pool 成向量。允许"100% 命中"会导致前向被完全跳过，采样器拿不到输入。
2. **投机解码留最后一块**：用 EAGLE/MTP 时，最后一块的 KV 会被 draft/verify 改写，不能从池子搬进来占住。
3. **粒度对齐**：只能按整块/整 chunk 复用（半块没有 key，也不是不可变对象）。

三刀的共同点是**往下压**命中数，所以工程规律是：

> **日志里看到的命中数，永远 ≤ 你算出来的理论值。差多少，就是这三刀里哪一刀砍的。**

### 2.7 线程模型：load 阻塞、save 异步

`KVPoolWorker` 启动时拉起后台线程（[`_start_kv_transfer_threads` L767+](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L767)）：

| 线程类 | 用在哪 | 语义 |
|---|---|---|
| `KVCacheStoreSendingThread` | 普通 bulk **save** | 异步 put，主线程不等 |
| `KVCacheStoreRecvingThread` | bulk **load**（`load_async=true`） | 异步 get |
| `KVCacheStoreLayerSendingThread` | **gva** layerwise save（Memcache） | 逐层 put |
| `KVCacheStoreKeyLayerSendingThread` | **block_key** layerwise save（Mooncake） | 逐 key put |
| `KVCacheStoreLayerRecvingThread` / `KVCacheStoreKeyLayerRecvingThread` | 对应 load | 逐层 get |

定义在 [kv_transfer.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L520)，基类 `KVTransferThread`。

**同步原语**是 per-layer 的 Event 数组：

```python
self.layer_load_finished_events = [threading.Event() for _ in range(self.num_layers)]
self.layer_save_finished_events = [threading.Event() for _ in range(self.num_layers)]
self.sync_save_events            = [torch.npu.Event()   for _ in range(self.num_layers)]
```

[L772-L775](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L772-L775)。`threading.Event` 管 CPU 侧线程同步，`torch.npu.Event` 管 device 侧"KV 真的写完显存了"——两个都要，缺一个就会读到没写完的数据（典型的"偶发精度问题"来源）。

**为什么 load 必须阻塞、save 可以异步？**
load 在关键路径上：数据没到，attention 就算错。save 不在：结果已经产出，put 失败最多是"下次不命中"，不影响本次正确性。这就是 `kv_load_failure_policy=recompute` 能成立的前提——**读失败可回退，写失败无所谓**。

### 2.8 五个"设计上的钉子"

1. **Connector 是薄壳**。看到 `AscendStoreConnector.xxx` 直接跳到 `KVPoolScheduler` 或 `KVPoolWorker` 的同名方法。
2. **命中永远要减 1**。所以"100% 命中"在这个系统里不存在。
3. **layerwise 强制 PIECEWISE 图模式**：`requires_piecewise_for_cudagraph` 直接返回 `use_layerwise`（[L98-L103](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L98-L103)）。因为逐层搬运需要在层之间插入同步点，FULL graph 抓不住。
4. **对象粒度和调度粒度是两回事**：scheduler 按 token/chunk 思考，backend 按 `(addr, size)` 描述符思考，中间由 `ReqMeta` 翻译。
5. **所有"多出来"的复杂度都在解耦**：PP（层在不同进程）、TP mismatch（head 切片不同）、hybrid（多 cache group）、DCP（上下文并行）——每一个都是"同一段 token 在不同并行拓扑下字节布局不同"，而 L3 的 key 就是用来吸收这些差异的。

### 2.9 改代码索引表

| 现象 | 第一个该看的文件:行 |
|---|---|
| 命中数是 0 / 比自己预期少 | [pool_scheduler.py:595](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L595)（先看走哪条分支） |
| 命中数对，但加载失败 | [pool_worker.py:3784](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3784) `lookup` / [kv_transfer.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py) recv 线程 |
| 启动就报配置错 | [connector:105-L133](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L105-L133)（参数互斥校验都在这里） |
| 传输慢 / TTFT 反而变差 | backend 的 `put/get` + layerwise 预取配置 |
| 精度/乱码，但不报错 | **几乎一定是 key 漏了维度**（L3）或 buffer 没注册（`register_kv_caches`） |
| 换 TP/PP 后命中归零 | L3 的 `infer_tp_mismatch_info` |

---

## 3. 关键问答记录

### 3.1 Q：第五节想让我学会什么？很多名词没基础，看不懂

**结论**：第五节原本只想传达一句话——

> **「命中数」是问出来的，而且问完之后还要被砍三刀；它不是一个简单的查表结果。**

"问谁"（三条分支）和"问的是什么"（block_key / gva）属于 **L2 和 L4**，在 L1 里是提前剧透，属于编排问题而非理解问题。

**名词人话对照表**

| 名词 | 人话 | 现在需要知道多少 |
|---|---|---|
| 池子 / store | 外部存 KV 的地方 | 知道它是"大仓库"即可 |
| **bulk** | 一个请求的所有层的 KV 打包成一批一起存、一起取 | 记住"打包" |
| **layerwise** | 每一层单独存、单独取 | 记住"拆开" |
| **block_key** | layerwise 时用"块的 hash + 层号"给每份数据贴标签 | L2 再讲 |
| **gva** | layerwise 时给统一编址内存里分配一段地址 | L2 再讲 |
| scheduler / worker | 前台（只有 CPU）/ 库房管理员（有仓库钥匙和显存地址） | 必须记住 |
| ZMQ / IPC | 同机两个进程之间的内部电话（`ipc://` 是本机管道） | 知道是"打电话" |
| `hbm_hit_tokens` | 本地显存里已算过、无需去池子取的长度 | 关键概念 |
| 裁剪 / trim | 算出来的数字不能直接用，还要往下压 | 本节重点 |
| EAGLE / MTP | 投机解码，会"借用"前缀最后一块 | 知道"有一种模式要留一块" |

**一个数字走完全流程**（`prompt=1000`，粒度 128，池子有前 6 块，本地已算 128）：

| 步骤 | 在干什么 | 结果 |
|---|---|---|
| ① 向下对齐 | 1000 不是 128 的整数倍，尾巴 104 个 token 凑不成整块 | 可查询长度 = **896** |
| ② 问池子 | "这 896 个 token 对应的前 7 块，你有几块？" → 前 6 块有 | 池子命中 = **768** |
| ③ 砍三刀 | 本例三刀都没触发 | 仍是 **768** |
| ④ 减掉本地已算 | 本地 128 已是现成的 | 需加载 = 768 − 128 = **640** |
| ⑤ 返回 | 告诉引擎"这 640 个 token 别算了" | `(640, ...)` |

**三条分支只是三种"问谁"的答案**（语义完全一样，都是问"从这个位置开始连续有多少 token 可用"）：

| 分支 | 什么情况下走 | 为什么必须这么问 |
|---|---|---|
| ① Mooncake + layerwise | 数据是"每层一个对象" | 要问"每一层都在吗"，需按层展开判断 |
| ② Memcache + layerwise | 数据是"每层一段地址" | 同上，但寻址方式不同 |
| ③ 普通 bulk（默认） | 数据是"整包一个对象" | 前台没有真仓库句柄，**打电话问 rank0 的管理员** |

**三刀的意义**：都是**往下压**命中数，所以"日志命中数 ≤ 理论值"，差值就来自这三刀。

**只带走三句话**：

1. **入口只有一个**：`get_num_new_matched_tokens()`（[pool_scheduler.py:595](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L595-L718)）。想知道"为什么没命中"永远从这里开始。
2. **命中数 = 池子命中 − 本地已算，再被三刀压**。分支只决定"问谁"。
3. **语义恒定**：无论 bulk/layerwise、Mooncake/Memcache，回答的都是"从这个请求的第 0 个 token 起，连续可复用多长"。

**学习顺序修正**：第五节是 L3/L4 的剧透。L1 只需留下「四层职责 + 8 个钩子 + 命中数会被砍三刀」；"问谁"的三条分支先当黑盒，等 L2 讲完两种存储语义，会发现它是**必然结果**而非设计选择。

### 3.2 Q：四层职责里的"Connector 空壳"没太理解

**用词纠正**：准确说法是「**薄壳 / 门面（Facade / Adapter）**」，不是"没代码"。它有 591 行，但**没有做任何业务判断**，只做三件事：

1. **接线**（`__init__`）：读配置、校验、按角色创建子对象；
2. **转发**（运行时）：vLLM 调它，它调子对象，**同名转发**；
3. **兜两个小工具类**：`AscendStoreKVEvents`、`LookupKeyServer`。

它的对立面不是"有逻辑的类"，而是**上帝类**——一个类把调度、搬运、存储全包了。

**为什么非存在不可：vLLM 只认一个名字**

```json
{ "kv_connector": "AscendStoreConnector", "kv_role": "kv_producer", ... }
```

vLLM 的工厂只登记**「名字 → 类」**一条映射（[kv_transfer/\_\_init\_\_.py L115-L119](vllm_ascend/distributed/kv_transfer/__init__.py#L115-L119)）：

```python
KVConnectorFactory.register_connector(
    "AscendStoreConnector",
    "...ascend_store.ascend_store_connector",
    "AscendStoreConnector",
)
```

于是 vLLM 的世界里只有这一个类名，它在两种时机各创建一次：

```
EngineCore 进程:  AscendStoreConnector(vllm_config, role=SCHEDULER)
每个 TP rank 进程: AscendStoreConnector(vllm_config, role=WORKER)
```

vLLM 完全不知道 `KVPoolScheduler` / `KVPoolWorker` 存在。所以这类必须是**唯一入口**：

```
      vLLM 认识的世界                    我们的世界
 ┌────────────────────────┐      ┌──────────────────────────────┐
 │ KVConnectorBase_V1     │      │ KVPoolScheduler   (查命中)    │
 │  · get_num_new_...     │ ───► │ KVPoolWorker      (搬字节)    │
 │  · start_load_kv       │      │ KVPoolBackend     (存到哪)    │
 │  · save_kv_layer ...   │      │                              │
 └────────────────────────┘      └──────────────────────────────┘
            ↑                                    ↑
       vLLM 只会调这个                  这些类 vLLM 一概不知
            └──────── AscendStoreConnector ─────┘
                          （接口在这里）
```

一次真实调用：

```
engine → AscendStoreConnector.get_num_new_matched_tokens()     ← 壳
       → KVPoolScheduler.get_num_new_matched_tokens()          ← 真正的命中计算
       → LookupKeyClient.lookup() 或 store_scheduler.batch_is_exist()
```

**唯一有逻辑的地方：`__init__`**

```python
self.backend_name = extra_config.get("backend", "mooncake").lower()
self.use_layerwise = extra_config.get("use_layerwise", False)
...
if role == KVConnectorRole.SCHEDULER:
    assert kv_cache_config is not None
    self.connector_scheduler = KVPoolScheduler(...)
else:
    self.connector_worker = KVPoolWorker(...)
    if not self.use_layerwise and rank == 0:
        self.lookup_server = LookupKeyServer(...)
```

[L105-L169](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L105-L169)。这里做的是：解析所有 `kv_connector_extra_config` 字段；**校验参数互斥**（如 `save_decode_cache` + `use_layerwise` 不允许，[L122-L133](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L122-L133)）；**按角色创建子对象**（"分裂成两个进程"的落地点）。

结论：**壳的"逻辑"只有一次性的接线校验，运行期零逻辑。**

**没有它会怎样（反证）**

| 方案 | 问题 |
|---|---|
| 让 vLLM 直接认识 `KVPoolScheduler` / `KVPoolWorker` | 得改 vLLM 上游工厂和配置格式；而配置里 `kv_connector` **只能写一个名字** |
| 让 `KVPoolScheduler` 自己继承 `KVConnectorBase_V1` | 那它必须同时实现 worker 侧方法（`register_kv_caches`、`start_load_kv`、线程模型…），但 scheduler 进程里**根本没有显存和 KV 张量** |

> **因为 vLLM 只给它一个接口，而实现住在两个进程里，所以中间必须有一个"只负责分派"的东西。**

**对读代码的 4 个实际用处**

1. **看栈就直接跳**：报错栈出现 `ascend_store_connector.py`，90% 真代码在下游**同名方法**。
2. **grep 会命中两处**：搜 `lookup_scheduler` 之类会同时命中壳和实现，**看实现那处**。
3. **壳上的方法列表 = 能力清单**：vLLM 的可选钩子（`get_finished`、`get_transfer_results`、`has_pending_push_work`）在壳上**有就是支持**（[L435-L512](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L435-L512)）。
4. **改功能的落点判断**：改行为 → Scheduler/Worker；改参数/校验 → 才在壳里。

**员工→总机类比**：公司对外只公布**一个总机号**，你拨过去说"找财务"，前台转接。前台不记账、不做决策，但它是**唯一入口**——没有它，客户手里就只有一个号。

### 3.3 Q：Backend 层的作用是什么

**一句话**：它把"把 KV 字节搬进/搬出池子"这件事，压缩成几个函数，让上层完全不知道底下是 Mooncake、Memcache 还是 Yuanrong。它是这个系统里的「**存储驱动层**」，对应 `JDBC driver` / Linux `VFS` / block driver 的位置。

**它隔离了三个"不稳定因素"**

| 不稳定因素 | 具体表现 | Backend 层怎么吸收 |
|---|---|---|
| **产品差异** | 三家不同 SDK，API 毫无共性 | 用一层 ABC 统一 |
| **物理模型差异** | 对象存储（key → object）vs 共享内存（地址 + 偏移） | 抽象成"key + 地址列表" |
| **能力差异** | 有的支持逐层传输、SSD 换出、租约/提交 | 用**可选方法**（默认 `NotImplementedError`）表达 |

**最小契约：6 个必需项**

```python
class Backend(ABC):
    store: Any | None = None
    requires_exists_before_put: bool = True      # 能力差异：put 前是否必须先过滤已存在的 key

    def __init__(self, parallel_config, lazy_init=False): ...
    @classmethod
    def create_scheduler_client(cls, parallel_config): ...   # 给 scheduler 用的轻量入口

    def set_device(self): ...
    def register_buffer(self, ptrs: list[int], lengths: list[int]): ...
    def exists(self, keys: list[str]) -> list[int]: ...
    def put(self, keys, addrs, sizes): ...
    def get(self, keys, addrs, sizes): ...
```

**① `register_buffer(ptrs, lengths)` —— 最关键**

```python
def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):   # worker 启动时调一次
```

必须先把显存地址"登记"给后端，因为**跨机搬运走 DMA（RDMA / UB / SDMA），不是 CPU 拷贝**，而这类硬件要求**目标内存预先注册（pin 住）**，否则网卡不允许直接写显存。这是 RDMA 语义的基本约束。**没注册过，一次都搬不了。**

**② `exists(keys) -> list[int]`** —— 命中查询的最底层，返回 0/1 列表。

**③ `put` / `get`** —— 注意签名是二维的：

```python
def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]])
```

> **一个对象 ≠ 一段连续内存。**

"第 3 块的 KV"在显存里可能是好几段不连续的区间：MLA 的多个分量、一个 chunk 里的多个物理页、TP mismatch 时按 head 切片的跨步区域……所以**一个 key 对应一组 `(addr, size)` 描述符**。

**第二组方法：layerwise 的会话式协议（全部可选）**

| 阶段 | 方法 |
|---|---|
| 写：划地 | `batch_alloc(keys, sizes)` → 返回一段地址 |
| 写：搬入 | `batch_copy_put(keys, buffers, sizes, dst_offsets)` |
| 写：宣告可见 | `batch_commit(keys)` |
| 读：准备 | `batch_get_start` / `batch_copy_get` / `batch_get_end` |
| 生命周期 | `batch_add_lease` / `batch_remove_lease` / `batch_revoke` / `batch_write_finish` / `batch_is_readable` |
| 能力声明 | `validate_layerwise_support()` |

**为什么 layerwise 需要一大套，而 bulk 只要一个 `put`？**

- **bulk（对象语义）**：`put(key, addr, size)` 一句话，后端自己负责拷完、自己管对象生命周期。
- **layerwise + GVA（共享内存语义）**：后端给的是**一段地址**，必须显式走完整套：先划地（alloc）、自己 DMA 进去（copy_put）、**告诉后端"我写完了"**（commit），别人才能读；还要租约（lease）防并发踩踏、revoke 回收。

> **对象存储替你管空间，所以接口简单；共享内存只给你地址，所以你必须自己管生命周期。**

**位置：一个层，两个入口**

```
                 ┌──────────────────────────┐
   Scheduler ───►│ create_scheduler_client()│──► 轻量 client（能 exists，不碰显存）
                 │                          │
   Worker    ───►│ Backend(...)             │──► 完整 backend（register_buffer + put/get）
                 └───────────┬──────────────┘
                             ▼
                  SDK: mooncake / memfabric / yuanrong
                             ▼
                     Mooncake Master / Memcache MetaService
```

```python
@classmethod
def create_scheduler_client(cls, parallel_config):
    return cls(parallel_config)          # 默认＝同一个类，只是不 register_buffer
```

[L82-L84](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py#L82-L84)；三个后端各自覆写了它（[mooncake:307](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L307)、[memcache:276](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py#L276)、[yuanrong:112](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/yuanrong_backend.py#L112)）。

这解释了 L1 里的疑问：为什么 scheduler 手里那个 `store_scheduler` 明明能 `batch_is_exist`，普通 bulk 的命中还要绕去 worker 问？因为**"能查"和"能搬"是两套权限**。

**边界：它不做什么**

| 不做 | 因为那属于 |
|---|---|
| 生成 key（block_hash、PoolKey） | `metadata.py`（L3） |
| 计算命中率、决定搬多少 | `KVPoolScheduler`（L1） |
| 重试、回退、失败策略 | `KVPoolWorker` + `kv_load_failure_policy` |
| 知道 request / block table / vLLM 是什么 | 谁都不该让它知道 |

它只认两样东西：**字符串 key** 和 **整数地址 + 长度**。保持"哑"，正是它的价值。

**加一个新后端只要 3 步**

1. 继承 `Backend`，实现那 6 个必需项；
2. 在 [backend/\_\_init\_\_.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/__init__.py#L22-L37) 的 `backend_map` 里注册一个名字：

```python
backend_map: dict[str, dict[str, Any]] = {
    "mooncake": {
        "name": "MooncakeBackend",
        "path": "...backend.mooncake_backend",
        "layerwise_protocol": "...backend.mooncake_layerwise",
    },
    "memcache": {
        "name": "MemcacheBackend",
        "path": "...backend.memcache_backend",
        "layerwise_protocol": "...backend.memcache_backend",
    },
    "yuanrong": { "name": "YuanrongBackend", "path": "...backend.yuanrong_backend" },
}
```

3. 配置里写 `"backend": "你的名字"`。

要支持 layerwise，再加那组 `batch_*` 和模块级 `LAYERWISE_DATA_PLANE = "block_key" | "gva"`——**这个字段就是 L2 的分水岭**（Mooncake layerwise 为 `block_key`，Memcache layerwise 为 `gva`）。

### 3.4 Q：`requires_exists_before_put` 为什么暴露在抽象层？

**它字面在干什么**：`save` 路径在真正 `put` 之前有一个过滤步骤：

```python
def _get_missing_indices(self, keys, require_exists_check=False):
    if not require_exists_check and not self.m_store.requires_exists_before_put:
        return list(range(len(keys)))          # ← 不查，全部当成"缺失"，直接全 put

    exists_states = self.lookup(keys)          # ← 先查一轮
    return [i for i, e in enumerate(exists_states) if not e]   # 只 put 不存在的
```

[kv_transfer.py L866-L875](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L866-L875)；layerwise 路径有同款逻辑（[pool_worker.py L2445-L2488](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2445-L2488)）。

两条控制流：

```
requires_exists_before_put = True（默认）
  keys ──► lookup(keys) ──► 过滤掉已存在的 ──► put(剩下的)
           多一轮 RPC                    少写重复数据

requires_exists_before_put = False（Yuanrong 是唯一一个）
  keys ──► put(全部)                       省掉那一轮 RPC
```

三个后端里只有 [yuanrong_backend.py L54](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/yuanrong_backend.py#L54) 声明了 `False`。

**为什么这个开关不能塞进 `put()` 内部？**

① **控制流原因**：它改变的是**调用方**的行为——是否多发一次 RPC；传给 `put` 的 key 列表**长度不同**；后续 KV event 里"哪些块真的被新写入"的记录不同（`_get_missing_indices` docstring 提到这点）。这些都不是 `put` 能单方面决定的。

② **性能原因**：这轮查询是**批量优化**。热点前缀场景下同一个块会被 N 个请求反复"想写"，先查一轮能省掉 N−1 次**整块 KV 的传输**（块是 16MB 量级）。但查一轮本身也有成本，是**带宽 vs 延迟**的权衡，取决于后端实现，调用方无法替后端决定。

③ **正确性原因**（最关键）：

```python
"""A block-key backend may reject overwriting a committed object: the put
session start fails for that key and a following range write can make
the readable object unusable. The layerwise save path must therefore
never open a session over a key that is already complete."""
```

[pool_worker.py L2446-L2452](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2446-L2452)

对某些后端，**覆盖一个已经 commit 的对象不是"幂等重写"，而是会把它弄坏**，而且破坏发生在 `put` 内部，等 `put` 返回失败已经来不及了——那个对象**此刻可能正被别的实例读取**。所以这条约束必须在调用方的控制流里、在 `put` **之前**执行。

**它暴露的"抽象不完美"是什么**

> **`put(key, ...)` 这个签名，假装"写入"在所有后端上是同一个动作；但真实语义有三种：幂等覆盖 / 允许覆盖但浪费 / 拒绝覆盖且可能破坏数据。**

抽象层只能吸收**能统一的部分**（key 是字符串、地址是整数、结果是成功/失败）。吸收不了的部分只有三种出路，全都是"泄漏"：

| 泄漏形式 | 本项目的例子 |
|---|---|
| **变成参数/开关** | `requires_exists_before_put` |
| **变成可选的"更高级 API"** | layerwise 那组 `batch_alloc / copy_put / commit` |
| **变成隐含的异常契约** | `BatchResultShapeError`、`put` 返回 `False` 表示检测到后端失败 |

**这就是"抽象不够完美"的事实**：不是设计者偷懒，而是**底层的物理语义本来就不一致**，任何抽象都只能把差异"挪个地方放"，不能消灭它。好的设计是**把差异显式化**（一个带 docstring 的布尔），坏的设计是让调用方去猜或去 `isinstance` 判断。

**同一个文件里还有第二个"同源泄漏"**

```python
def batch_is_exist(self, keys: list[str]) -> list[int]:
    return self.exists(keys)          # 默认：循环调用单条
```

[base.py L102-L103](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py#L102-L103)。三个后端都覆写了它，直接转给 SDK 的真批量接口（[mooncake:339](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L339)、[memcache:324](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py#L324)、[yuanrong:132](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/yuanrong_backend.py#L132)）。它和本题是同一类东西：`exists` 是**必须**（语义），`batch_is_exist` 是**可加速**（性能）。

所以这一层的结构是：**最小公共集（6 个抽象方法）+ 一堆可选加速/扩展**。

**工程上怎么处理这种泄漏（三条惯例）**

1. **显式契约 + 文档**：在抽象类里写成带注释的属性。
2. **默认值选保守（安全但慢）**：默认 `True`，让快的后端主动 `opt-out`。默认选错方向的代价是数据损坏。
3. **用 `getattr(..., True)` 读取**：

```python
if not getattr(self.m_store, "requires_exists_before_put", True):
    return keys
```

[L2463](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2463)。因为 `m_store` 可能是 mock、可能是没实现该属性的老后端——用默认值兜底保证向后兼容。这本身就是"契约只能靠约定、不能靠类型系统"的症状。

**标准答案版**

> `requires_exists_before_put` 暴露在抽象层，是因为"重复写同一个 key"在不同后端上语义不同（幂等覆盖 / 可覆盖 / 拒绝且可能破坏数据），而这一差异决定了**调用方要不要在 put 之前多做一轮 exists 过滤**——这是控制流层面的决定，无法在 `put` 内部完成。它暴露的事实是：**`put` 这个统一签名掩盖了底层写入语义的不一致，抽象只能把差异显式化成参数，不能消灭它。**

**延伸：接入第四个后端（如本地 NVMe）该选 True 还是 False**

三连问：

1. 重复写同一个 key 会**报错**吗？
2. 重复写会**破坏正在被读的数据**吗？
3. 重复写的**带宽代价**大吗（每块 16MB）？

任一为"是" → 选 `True`；三个都为"否"（实现天然幂等、覆盖零成本）→ 才能选 `False`。选 `False` 等于**承诺一个契约**——"我的 put 是幂等且安全的"，这个承诺没有编译器检查，只有测试和 review。

---

## 4. 实操命令 / 代码

### 4.1 本讲涉及的代码块索引

| 代码块 | 位置 | 用途 |
|---|---|---|
| 角色分裂 | [connector L153-L169](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L153-L169) | 说明"同一个类实例化两次" |
| 三行转发 | [connector L235-L248](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L235-L248) | 证明 Connector 是薄壳 |
| `__init__` 校验与分发 | [connector L105-L169](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L105-L169) | 壳里唯一有逻辑的地方 |
| `requires_piecewise_for_cudagraph` | [connector L98-L103](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L98-L103) | layerwise 强制 PIECEWISE |
| `LookupKeyServer` REP 循环 | [connector L513-L584](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L513-L584) | bulk 命中查询的 ZMQ 应答端 |
| 命中三刀 | [pool_scheduler L674-L679](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L674-L679) | 命中数为什么被压 |
| 分支选择 | [pool_scheduler L612-L647](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L612-L647) | 三条命中查询分支 |
| per-layer 同步原语 | [pool_worker L772-L775](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L772-L775) | `threading.Event` + `torch.npu.Event` |
| `_get_missing_indices` | [kv_transfer L866-L875](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L866-L875) | `requires_exists_before_put` 的两条控制流 |
| `_filter_pool_existing_keys` docstring | [pool_worker L2445-L2488](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2445-L2488) | 覆盖已 commit 对象会破坏数据的证据 |
| `Backend` ABC | [base.py L73-L175](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py#L73-L175) | 最小契约 + 可选协议 |
| `create_scheduler_client` | [base.py L82-L84](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py#L82-L84) | 一个层两个入口 |
| `backend_map` 注册表 | [backend/\_\_init\_\_.py L22-L37](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/__init__.py#L22-L37) | 换存储只改这里 + 配置 |
| `LAYERWISE_DATA_PLANE` | [mooncake_layerwise.py L27](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_layerwise.py#L27) / [memcache_backend.py L83](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py#L83) | `block_key` vs `gva` 的分水岭 |

> 可运行的验证命令（mock 化 UT）见 `00-学习路线.md` 的「8.2 可运行验证」一节。

---

## 5. 踩坑点、报错与解决方案

| 坑 | 现象 | 根因 | 解决 / 定位 |
|---|---|---|---|
| **换了 TP 后命中率归零** | 无报错，收益突然消失 | TP mismatch 未启用（MLA / hybrid / 不整除 / heads 不足），key 空间静默退化 | 查 [infer_tp_mismatch_info](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L80-L113) 的 `enabled` 条件；入口仍是 `get_num_new_matched_tokens` |
| **静默错数据（精度/乱码，不报错）** | 结果异常但无异常栈 | key 漏了维度（TP/PP/group/layer/cache_role…），或 buffer 没注册 | 对比 `PoolKey.to_string()` 的字段是否覆盖全部分片维度；检查 `register_kv_caches` 是否执行 |
| **覆盖已 commit 对象** | put 失败，或对象变得不可读 | block-key 后端拒绝覆盖已提交对象 | 保持 `requires_exists_before_put=True`，save 前先过滤（[L2446-L2452](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2446-L2452)） |
| **池子故障导致请求失败** | 池子不可用时服务报错 | `kv_load_failure_policy=fail` | 池子定位是"可失败的加速器"，应配 `recompute` |
| **layerwise 下 graph 抓取失败** | 启动/捕获阶段报错 | layerwise 需要层间同步点 | 走 PIECEWISE（`requires_piecewise_for_cudagraph` 返回 `use_layerwise`） |
| **命中数低于预期** | 理论值 768，日志更小 | 三刀：不能全命中 / EAGLE 留块 / 粒度对齐 | 逐条核对 [L674-L679](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L674-L679) 的三个条件 |
| **抽象泄漏导致旧后端 AttributeError** | 访问 `requires_exists_before_put` 报错 | mock/老后端没有该属性 | 用 `getattr(..., True)` 兜底 |
| **"为什么这里还有一个同名方法"** | 读代码困惑 | Connector 与下游同名转发 | 直接跳到 `KVPoolScheduler` / `KVPoolWorker` 的同名方法 |

---

## 6. 小结 & 待验证点

### 6.1 小结

1. **四层职责**：Connector 薄壳（唯一入口、只做接线与转发）→ Scheduler 算命中与产出清单 → Worker 注册显存并搬字节 → Backend 提供存储语义。
2. **8 个钩子**构成一次请求的完整骨架；其余代码都在处理边界情况。
3. **命中数 = 池子命中 − 本地已算，再被三刀压**；分支只决定"问谁"，语义恒定为"从第 0 个 token 起连续可复用多长"。
4. **load 阻塞、save 异步**，因为读错会算错、写错只会少一次命中。
5. **抽象只能把差异显式化**：`Backend` 的 6 个必需方法 + 可选协议 + `requires_exists_before_put` 这类开关，都是底层语义不一致的显式化结果。

### 6.2 待验证点

1. **L1 的 5 道自检题未作答**：
   1. 为什么命中等于全 prompt 长度时要减 1？pooling（embedding）模型还需要减吗？
   2. bulk 模式下命中查询为什么要绕到 worker 进程？`store_scheduler` 为什么不够用？
   3. save 的 put 失败 / load 的 get 失败分别会发生什么？各由哪个配置兜底？
   4. `wait_for_layer_load` 是每层调一次还是一次？对 graph 模式有什么要求？
   5. 要接入一种全新存储，只需改哪一层？要实现哪几个方法？
2. **Backend 自检 3 题中 2 题未答**：
   1. 为什么 `register_buffer` 必须存在？用 CPU memcpy 行不行？
   2. `put` 的 `addrs` 为什么是二维的？举一个"一个 key 多段地址"的已见场景。
3. **未在真机验证**：L1 全部结论来自代码阅读与文档，尚未在 NPU + Mooncake/Memcache 环境实测。
4. **未运行 mock 化 UT**（命令见 `00-学习路线.md` 8.2），尚未用改动数字的方式验证断言行为。
5. **第五节是否改写为 L2 引子**（曾提议：把三条分支挪到讲完两种存储语义之后），**未答复**。
6. **L2 尚未开始**，`block_key` 与 `gva` 两种数据面的具体差异仍为黑盒。