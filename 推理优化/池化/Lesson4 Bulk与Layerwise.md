# 04 · Lesson 4 知识点：Bulk vs Layerwise 数据通路

> 读法：先读 §0 全景（一张图 + 六条要点），再进正文。§9 是答疑沉淀（save/load 语义、Mooncake vs Memcache、池化与部署形态），读完正文后再看。
> 文件路径相对仓库根 `vllm-ascend`，行号于 2026-11-02 核对。
> 先修：L0（KV 有多大）、L1（8 个钩子）、L2（二维地址、两套仓库接口）、L3（粒度、key）。

---

## 0. 全景：两条数据通路

### 0.1 结构框图

```
        一个请求命中 6 块（6 × 22 MB = 132 MB），要搬进/搬出池子
                                 │
        ┌────────────────────────┴─────────────────────────┐
        ▼ ① bulk（默认，use_layerwise=false）                ▼ ② layerwise（use_layerwise=true）
┌────────────────────────────┐              ┌─────────────────────────────────────┐
│ 一次调用搬完 132 MB          │              │ 切成 44 份，每层一份                  │
│ start_load_kv → recv 线程   │              │ wait(0) 算0 wait(1) 算1 … wait(43)   │
│   → get(keys, addrs, sizes)│              │ save(0) save(1) … save(43)          │
│ wait_for_save → put        │              │ 等首层；其余藏在计算背后               │
│ 等待 = 全量                 │              │ 等待 = 第一层（1/44）                 │
└────────────────────────────┘              └──────────────┬──────────────────────┘
                                                           │ 落地方式由后端声明
                        ┌──────────────────────────────────┼──────────────────────────────────┐
                        ▼                                  ▼                                  ▼
              ┌───────────────────┐            ┌───────────────────┐            ┌──────────────────┐
              │ block_key(Mooncake)│            │ gva(Memcache)     │            │ 没声明（历史路径） │
              │ put_start         │            │ alloc             │            │ 每层直接 put      │
              │ copy_put(偏移)     │            │ copy(GVA)         │            │ 每层一个对象       │
              │ commit            │            │ write_finish      │            │ 线程：KeyLayer*   │
              └───────────────────┘            └───────────────────┘            └──────────────────┘
              线程：LayerSending/RecvingThread（key-major 分支）                   线程：KeyLayerSending/Recving
```

### 0.2 这三块分别在说什么（三句话）

1. **bulk 是"整箱快递"**：一次函数调用搬完一个请求所有层的 KV；简单，但等待是全量的。
2. **layerwise 是"流水线传送带"**：把搬运切在**层边界**上，第 0 层到了就先算，后面几层在路上；代价是 44 次小传输 + 层间同步。
3. **layerwise 只是"切时间"**，切完用哪套仓库接口搬，由后端的 `LAYERWISE_DATA_PLANE` 决定——这是 L2 §6 讲的三套做法。

### 0.3 读图要点（6 条）

1. **两条通路共用同一套 key / addrs**，区别只在"一次搬多少"和"什么时候搬"。
2. **`use_layerwise` 下三个逐层钩子才"活"**：bulk 里 `wait_for_layer_load` / `save_kv_layer` 直接 `return`。
3. **同步要两套**：`threading.Event` 管 CPU 线程、`torch.npu.Event` 管"KV 真的写进显存了"，缺一个就会读到半写完的数据。
4. **预取窗口 = `num_prefetch_layers`**（默认 `min(shared_buffers, 8)`），领先太多会抢 attention 的 HBM 带宽。
5. **layerwise 只能 Prefill**，且必须 PIECEWISE 图；与 `save_decode_cache` 互斥。
6. **收益是"隐藏等待"不是"加速传输"**：首层等待从 1.4 GB 降到 32 MB，代价是 44 次小 DMA 的启动开销。

### 0.4 图 ↔ 章节对照

| 图里的位置 | 本文哪一节 | 一句话 |
|---|---|---|
| 两条通路的分叉 | §1 | 用 1.4 GB 算一遍"等待暴露在哪里" |
| bulk 那半边 | §2 | 一条函数调用 + 一个后台线程 |
| layerwise 那半边 | §3 | 44 个同步点、两套 Event、预取窗口 |
| 三套落地方式 | §4 | 线程选择代码 + 会话时间线 |
| 约束（prefill / PIECEWISE） | §5 | 为什么不能默认开 |
| 排障 | §6 | 开了 layerwise 却像每层一个对象 |

### 0.5 先补三个概念

**(1) 层边界**：模型前向是 `for layer in 0..43` 的循环。layerwise 把"搬"插在这个循环的每个空档里——所以它不需要额外的线程模型，只需要在循环里回调。

**(2) 重叠（overlap）**：DMA（搬数据）和 NPU 计算可以同时进行，**但它们抢同一份 HBM 带宽**。所以"提前搬"不是免费的：搬得太早太多，会让 attention 变慢。

**(3) 图模式**：FULL graph 把整个 forward 抓成一张图一次性提交，**中间没有 Python 回调点**；PIECEWISE 按段提交，段与段之间可以插入同步。layerwise 需要在每层之间回调，所以必须 PIECEWISE。

---

## 1. 为什么需要两条通路：把等待算出来

用 L0/L3 的数字：44 层、8 KV head、head_dim 128、BF16、block=128 → **一个 block 22 MB**，8K token 请求 = 64 块 = **1.4 GB**。

```
bulk 的 load：      [=========== 搬 1.4 GB ===========]  层0  层1 ... 层43
                      ↑─────────────────────↑
                      这段时间 attention 只能干等（100% 暴露在 TTFT 里）

layerwise 的 load： [搬层0] 层0  [搬层1] 层1  [搬层2] 层2 ...
                      ↑等 32 MB
                                ↑层1 已就绪，立即返回
```

- 首层等待：**1.4 GB → 32 MB（1/44）**
- 尾部：bulk 的 `put` 在 forward 结束后才开始；layerwise 的第 0 层在层 0 算完就开始写，等 forward 结束只剩最后几层在飞。

**结论**：layerwise 换的不是带宽效率（反而更差），换的是**延迟**。所以它只在"传输时间相对计算时间不可忽略"时才划算。

---

## 2. Bulk 通路：一条调用 + 一个线程

### 2.1 load

```
start_load_kv(metadata)
   └─ KVPoolWorker.start_load_kv → 把任务塞进 KVCacheStoreRecvingThread
        └─ get(keys, addrs, sizes)          ← 一次搬完所有层（L2 的二维地址）
wait 完成后 forward 才能开始
```

`load_async=true` 时是异步 get，但引擎仍然要等到"本步需要的 KV 到齐"才进 forward。

### 2.2 save

```
forward 结束 → wait_for_save(metadata)
   └─ KVCacheStoreSendingThread：put(keys, addrs, sizes)  → 后台线程
   主线程不等（save 失败只影响"下次不命中"）
```

### 2.3 两个特点

| | bulk |
|---|---|
| 优点 | 1 次大传输、DMA 效率最高；实现简单；不限制图模式 |
| 缺点 | 等待全量暴露；峰值缓冲要装下整包 |
| 适用 | 大多数场景（默认）、Decode 侧写回、以及 layerwise 不支持的地方 |

---

## 3. Layerwise 通路：把 8 个钩子变成 44 个同步点

### 3.1 钩子的开关（connector）

```python
def wait_for_layer_load(self, layer_name):
    if not self.use_layerwise: return          # ← bulk 下是空函数

def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kw):
    if not self.use_layerwise: return

def wait_for_save(self):
    if self.use_layerwise: return              # ← 债已在逐层还完
```

[ascend_store_connector.py L368-L372 / L412-L416 / L424-L433](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L412-L433)

### 3.2 两套同步原语（这张表是本节的核心）

[pool_worker.py L771-L775](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L771-L775)：

```python
self.layer_load_finished_events = [threading.Event() for _ in range(num_layers)]   # CPU 侧
self.layer_save_finished_events = [threading.Event() for _ in range(num_layers)]   # CPU 侧
self.sync_save_events           = [torch.npu.Event()  for _ in range(num_layers)]   # device 侧
```

| 原语 | 管什么 | 用在哪 |
|---|---|---|
| `threading.Event` | 传输线程"这层搬完了" | `wait_for_layer_load` 里等 |
| `torch.npu.Event` | **KV 字节真的写进显存了** | `save_kv_layer` 先 `record()`，发送线程再 `synchronize()`（[kv_transfer.py L2067](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L2067)） |

> `save_kv_layer` 是 **Python 侧**被调用的，此时 NPU 上的 attention kernel 可能还没执行完。所以必须先记一个 device event，让 DMA 等它——**少这一个，就是"偶发精度问题"**。

### 3.3 预取窗口

[pool_worker.py L3083-L3108](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3083-L3108)：

```python
window_end = min(self.current_layer + self.num_prefetch_layers, self.num_layers - 1)
while self.next_layer_to_submit <= window_end:
    submit_layer_load(...)          # 提前把未来几层的 load 任务塞给接收线程
```

`num_prefetch_layers` 默认 `min(shared_buffers, 8)`（[layerwise_cache_layout.py L29](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_cache_layout.py#L29) `_DEFAULT_MAX_PREFETCH_LAYERS = 8`），可用 `layerwise_prefetch_layers` 调。

### 3.4 一次 load 的完整时序

```
for layer in 0..43:
    wait_for_layer_load()          # ① 提交未来窗口内的任务 ② 等本层 Event（10s 超时打日志）
    attention(layer)               # ③ 计算
    save_kv_layer()                # ④ record npu event → 把本层任务丢给发送线程
```

`wait_for_layer_load` 里那句循环（[L3135-L3137](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3135-L3137)）：

```python
while not self.layer_load_finished_events[self.current_layer].wait(timeout=10):
    self.kv_recv_thread.raise_if_failed()
    logger.info("Layerwise %d load not done, keep waiting", self.current_layer)
```

**超时 10 秒不是失败**，只是打一行日志继续等——因为传输慢不等于传输错。真正的失败靠 `raise_if_failed()` 暴露。

### 3.5 attention fence：为什么预取要等"计算真的开始"

[attention_fence.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/attention_fence.py)：attention 提交前 `record` 一个 NPU event，预取的 DMA 等它完成才提交。

```
attention 快开始 → record event → (DMA 线程等到) → 提交 H2D/L2G
```

理由：**DMA 和 attention 抢同一份 HBM 带宽**。无脑抢先发，会把这一层的计算拖慢。overlap 是有代价的，gate 是"让传输等计算真的起来了再插进来"。

---

## 4. 三套落地方式：切完之后怎么搬

### 4.1 线程选择代码（唯一判据）

[pool_worker.py L771-L901](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L771-L901)：

```python
if (self.use_block_key_layerwise or self.use_layerwise_transfer) and can_save:
    self.kv_send_thread = KVCacheStoreLayerSendingThread(...)      # block_key / gva
elif self.can_put:
    self.kv_send_thread = KVCacheStoreKeyLayerSendingThread(...)   # 历史路径
...
if self.use_block_key_layerwise or self.use_layerwise_transfer:
    self.kv_recv_thread = KVCacheStoreLayerRecvingThread(...)
else:
    self.kv_recv_thread = KVCacheStoreKeyLayerRecvingThread(...)
```

其中（[pool_worker.py L241-L242](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L241-L242)）：

```python
self.use_block_key_layerwise = self.use_layerwise and self.layerwise_data_plane == "block_key"
self.use_layerwise_transfer  = self.use_layerwise and self.layerwise_data_plane == "gva"
```

> **判据只有一条**：后端模块有没有声明 `LAYERWISE_DATA_PLANE`。没声明 → 两个布尔都是假 → 悄悄走历史路径，**没有任何告警**。

### 4.2 对照表

| | `block_key`（Mooncake） | `gva`（Memcache） | 没声明（历史） |
|---|---|---|---|
| 一个对象 | 1 block × 1 head 的**整层栈** | 同左，一段 GVA 区域 | **1 block × 1 head × 1 层** |
| key | `模型@{hash}@{head}` | `模型@[group]@{hash}@{head}` | 多一个 `@layer_id` |
| 写 | `put_start` → 逐层 `copy_put(偏移)` → `commit` | `alloc` → 逐层 `copy(GVA)` → `write_finish` | 每层直接 `put` |
| 读 | `get_start` → 逐层 `copy_get` → `get_end` | `add_lease` → 逐层 `copy(G2L)` → `remove_lease` | 每层直接 `get` |
| 线程 | `LayerSendingThread`（key-major） | `LayerSendingThread`（GVA 分支） | `KeyLayerSendingThread` |

### 4.3 一次保存的时间线（两套几乎同形）

```
[对象语义 / block_key]                      [地址语义 / gva]
put_start(6 keys, [22MB]*6)                  alloc(6 keys) → 6 个 GVA
   │                                            │
层0：等 sync_save_events[0] → copy_put(偏移0)   层0：等 event → copy(GVA+偏移)
层1：…                                          层1：…
层43：copy_put(最后一层)                        层43：copy(最后一层)
   │                                            │
commit(6 keys) → 此刻才可见                     write_finish(6 keys) → 此刻才可读
失败：revoke(该 key)                            失败：停止本步（不删别人的 key）
```

代码在 [kv_transfer.py L2059-L2112](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L2059-L2112)：逐层 `batch_copy_put`，最后一层触发 `batch_commit`，失败 key 走 `revoke`。

### 4.4 两个细节

- **`use_key_major_ranges`**：同一个发送线程内部还有两条分支（[L2161](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L2161)、[L2455](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L2455)）。key-major 是"这次任务覆盖哪些 key、每 key 哪些段"（block_key/gva 用），否则走老的"一层一个任务"。
- **历史路径按 `put_step` 跨步**（[L1819-L1821](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1819-L1821)）：`starts = starts[tp_rank % put_step :: put_step]`——当 `num_kv_heads < tp_size` 时，多个 TP rank 的 KV 是**复制的**，只让每隔 `put_step` 个 rank 去写，避免重复写同一对象。

---

## 5. 约束与代价

| 约束 | 证据 | 为什么 |
|---|---|---|
| **只能 Prefill** | 官方文档：`use_layerwise` "only supported on the Prefill node" | decode 每步只算 1 个 token，44 次小 DMA 的启动开销 > 收益 |
| **必须 PIECEWISE 图** | `requires_piecewise_for_cudagraph` 直接 `return use_layerwise`（[connector L98-L103](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L98-L103)） | FULL graph 中间没有 Python 回调点 |
| **与 `save_decode_cache` 互斥** | 显式报错并给文案（[L122-L133](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L122-L133)） | 该模式要求非 layerwise |
| **小传输更多** | 44 次 vs 1 次 | DMA 启动开销、描述符数量 |
| **抢带宽** | `attention_fence` | 搬太早会让本层计算变慢 |
| **partial chunk 语义更细** | 文档：partial layerwise 只拷每段覆盖的 token 区间 | 命中仍按块对齐，但传输可以更细 |

### 什么时候"必须"用 layerwise

1. **整包等待扛不住**：前缀很长（几十万 token）、带宽有限 → bulk 的首层等待直接吃掉 TTFT。
2. **要尽早发布**：P 节点边算边写，D 侧能更早取到前若干层（PD 分离场景）。
3. **显存不够放整包缓冲**：layerwise 只需要几层的环形缓冲。
4. **后端只有地址平面**：`gva` 的完整能力（租约、逐层可见）本身就依赖 layerwise。

反过来：**吞吐优先、前缀短、带宽充裕 → 用 bulk**（默认）。

---

## 6. 排障表

| 现象 | 原因 | 先看哪 |
|---|---|---|
| 开了 `use_layerwise`，行为却像"每层一个对象" | 后端没声明 `LAYERWISE_DATA_PLANE` → 静默走历史路径 | 实际创建的是哪个线程类 |
| layerwise 下图捕获失败 | 没有落到 PIECEWISE | `requires_piecewise_for_cudagraph` 的返回 |
| Decode 节点开 layerwise 报配置错 | 只能 Prefill | connector 的互斥校验 |
| 偶发精度异常、无栈 | `sync_save_events` 没记录 / commit 前就被读 | `save_kv_layer` 与 `commit` 时序 |
| 传得快但 TTFT 没改善 | 预取窗口太大 / 抢占带宽 | `layerwise_prefetch_layers`、`attention_fence` |
| 30B 以上模型命中率异常 | 块大小/多头组合下描述符或分区差异 | 各 group 的 `block_size`、`put_step` |

---

## 7. 自检 5 题

1. 8K token、64 块、每块 22 MB：bulk 与 layerwise 的"首层等待"分别是多少？为什么 layerwise 的带宽效率反而更低？
2. `threading.Event` 与 `torch.npu.Event` 各解决什么问题？只留一个会出什么现象？
3. 一个后端模块里没有 `LAYERWISE_DATA_PLANE`，但配置了 `use_layerwise=true`。会发生什么？你怎么从日志/线程类型判断出来？
4. 为什么 layerwise 必须 PIECEWISE？FULL graph 具体缺了什么？
5. 什么情况下你会主动从 bulk 切到 layerwise？举两个场景，并说明各自的代价。

---

## 8. 待验证点

1. 全部结论来自源码阅读 + 官方文档，**未在 NPU 真机验证**。
2. §3.3 的预取窗口默认值来自 `_DEFAULT_MAX_PREFETCH_LAYERS = 8`；`shared_buffers` 的实际取值路径未追踪。
3. §4.4 历史路径 `put_step` 跨步的动机（`num_kv_heads < tp_size` 时 KV 复制）是**推断**，未找到注释原文。
4. layerwise 对 TTFT 的实际收益未做实测，§1 只有量级估算。

> **2026-11-02 追加核对**：§4.2 的 key 格式与两个后端的 `LAYERWISE_DATA_PLANE` 声明已在源码中确认（见 §9.2 对照表）；第 2 条的 `shared_buffers` 取值路径仍未追；第 4 条的 TTFT 收益仍未实测，§9.4 只做了定性推导。

---

## 9. 答疑沉淀：把 L4 放回整个系统里

> 本节记录 2026-11-02 讨论中反复卡住的几个点，按"先给结论、再给证据"写。
> §9.1–§9.3 补 L4 的术语与边界，§9.4–§9.6 讨论池化在部署里的位置——**超出 L4 原文范围**，属于横向延伸。

### 9.1 save / load 到底指什么

| 操作 | 方向 | 钩子 | 底层动作 | 线程 | 在关键路径上？ |
|---|---|---|---|---|---|
| **load** | 池子 → 本机 HBM | `start_load_kv`（bulk）<br>`wait_for_layer_load`（layerwise） | `get` / `get_start`+`copy_get`+`get_end` / `add_lease`+`copy`+`remove_lease` | `KVCacheStoreRecvingThread` | **是**，到齐才进 forward |
| **save** | 本机 HBM → 池子 | `wait_for_save`（bulk）<br>`save_kv_layer`（layerwise） | `put` / `put_start`+`copy_put`+`commit` / `alloc`+`copy`+`write_finish` | `KVCacheStoreSendingThread` | **否**，主线程不等 |

三个易错点：

1. **命名视角是"计算节点"**：load 是"把 KV 装进设备"，不是加载模型权重；save 是"写进远端池子"，不是落盘（落盘是 SSD Offload 那一层的事）。
2. **`load_async=true` 不等于 load 完全异步**：get 异步提交，但引擎必须保证"本步要用的 KV 已到齐"才进 forward。
3. **失败语义不对称**：load 失败 → 该请求退化为不命中（`_record_failed_request`，[pool_worker.py L1431](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1431)）；save 失败 → 只影响"下次不命中"，当前结果不受影响。

代码锚点：[ascend_store_connector.py L347/L368/L412/L424](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L347)、[pool_worker.py L1374](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1374) / [L3265](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3265)。

### 9.2 Mooncake 与 Memcache 是什么

两者都是 `AscendStoreConnector` 背后的**池子存储引擎**（`kv_connector_extra_config.backend`，默认 `mooncake`；第三个后端是 `yuanrong`）。它们解决同一个问题：NPU 显存放不下的 KV 放哪、怎么取回来。

**一句话比喻**：Mooncake 是"分布式仓库 + 单号"（对象语义），Memcache 是"共享停车场车位"（地址语义）。

| | **Mooncake** | **Memcache** |
|---|---|---|
| 出身 | Moonshot AI 开源，Kimi 的服务底座 | 华为 MemFabric 之上的池化共享内存；**不是 `memcached`** |
| 组件 | Transfer Engine（TE）+ Mooncake Store（master 管元数据）+ Conductor | MetaService（元数据）+ LocalService（每节点贡献 DRAM）+ 客户端 |
| 抽象 | 对象（key → blob） | 地址（GVA 区域 + 偏移） |
| 数据平面 | `block_key`（[mooncake_layerwise.py L27](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_layerwise.py#L27)） | `gva`（[memcache_backend.py L83](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py#L83)） |
| 写 | `put_start` → 逐层 `copy_put` → `commit`；失败 `revoke` | `alloc` → 逐层 `copy` → `write_finish`；失败停手 |
| 读 | `get_start` → `copy_get` → `get_end` | `add_lease` → 逐段 `copy` → `remove_lease` |
| key | `model@hash@rank`；hybrid 走 layout+group（[L83-L91](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_layerwise.py#L83-L91)） | `model@hash@rank`；多 group 才加一段（[L117-L135](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py#L117-L135)） |
| 分层 | DRAM / **SSD Offload** | HBM → 二级池化 DRAM → 三级池化 DRAM → SSD |
| 额外能力 | 租户配额、Store/PD 流量分离、metrics | A3 HCCS/UB、`device_sdma`、MemCache 与 vLLM 分离部署、DP init barrier |
| 在仓库里的角色 | 既能做 P/D 搬运工（`MooncakeConnector*`），也能做池子 | **只做池子**（KV 池 + 多模态 EC cache） |
| layerwise 版本门槛 | ≥ 0.3.14 | `memcache_hybrid` ≥ 1.3.0 |

> ⚠️ 别把两件事混起来：`MooncakeConnectorV1/V2`（P/D 之间的 P2P 搬运）和 `MooncakeBackend`（池子后端）是**两套东西**，§4 讲的是后者。

### 9.3 "池化只作用于 prefill"要分两层

- **默认配置**确实只在 P 侧做 put/get：D 节点通过 P2P connector 拿 KV（[KV_Cache_Pool_Guide.md L50-L52](docs/source/developer_guide/Design_Documents/KV_Cache_Pool_Guide.md#L50-L52)）。理由是避免 P→D 和 P→池子两条数据流抢带宽。
- **机制上 decode 也能写**：`save_decode_cache`、`consumer_is_to_put`、`consumer_is_to_load` 三个开关都在（[pool_scheduler.py L111-L125](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L111-L125)），只是默认关——decode 每步只算 1 token，收益低。
- 真正 prefill-only 的是 **`use_layerwise`**（§5），别把它扩大到整个池化。

### 9.4 PD 分离下池化的作用是否更大

**收益 ≈ 命中前缀 token 数 × 每 token prefill 成本 × 命中次数 −（put + get + 带宽竞争）**

PD 分离放大的是"命中次数"里的**实例数因子**，而且是间接的：

1. **P 层无状态化**：P 做成可扩缩容的副本后，本地 HBM 里的 prefix cache 随实例上下线消失，池子成为唯一稳定的沉淀层——这是最硬的理由。
2. **P/D 解耦**：P2P 是"算完推给某个 D"，池子是"commit 后任何 D 随时取"，支持跨时间复用。
3. **长上下文/大模型**：单个 token 的 prefill 成本高，命中一次的绝对收益更大。

但"更大"不等于"白得"：

| 代价 | 说明 |
|---|---|
| 双份带宽 | P 本来就要 P2P 推给 D，再 put 进池子就是第二条流；A3/950 上要做 Store/PD 流量分离（`ASCEND_GLOBAL_RESOURCE_CONFIG`） |
| P 侧 TTFT 更敏感 | put 异步不阻塞，但 **get 同步**；命中率不高时"查池 + 回读"的固定开销直接进 TTFT |
| 前缀重复率 | 一次性语料 / 生成型负载命中率≈0，纯亏 |
| PD 混合被低估 | 共置部署没有双份带宽问题，跨请求复用照样有效 |

**真吃到红利**：前缀重复率高（系统 prompt / few-shot / 多轮）、P 实例数 > 1 且请求被摊开、上下文长（万级 token 起）、有容量兜底、带宽有余量。
**别开**：单 P 实例且本地装得下、前缀几乎不重复、短 prompt、带宽已吃紧。

### 9.5 "P 实例"是什么，跟会话数无关

| 概念 | 是什么 | 由什么决定 | 影响池化哪一项 |
|---|---|---|---|
| **P 实例数** | 一次 `vllm serve` 起的、独占一组 NPU 卡的副本；有自己的模型权重副本 + KV 显存 + prefix cache | 部署规模（副本数、DP 拆几个） | **跨实例复用能否发生** |
| **会话数/并发** | 同时在线的请求流 | 业务流量 | 命中次数、带宽竞争 |

**不是**：一个用户会话、一个请求、一张卡（TP=8 的一个实例占 8 张卡）。

```
              负载均衡（LB）
        ┌──────────┴──────────┐
        ▼                     ▼
  P 实例 #1  ──┐         P 实例 #2  ──┐   ← 两块"冰箱"，互相看不见
  (4 卡, KV 60G)│        (4 卡, KV 60G)│
        └───────┴─────┬───────┴───────┘
                      ▼
                  [ KV 池子 ]           ← 唯一共享的"中央仓库"
```

"跨实例"实际是"跨 KV cache 所有者"，包括四类：多副本、同一实例内的多个 DP rank（各有独立 scheduler 和 block manager）、同一 rank 重启前后、本地 prefix cache 被 evict 之后。

两个反直觉点：

- **单实例 ≠ 不用池化**：前缀长、并发高、本地装不下时，池子就是二级缓存。
- **会话数多 ≠ 池化收益大**：若会话粘在同一副本，上一轮 KV 就在本地 HBM 里，走的是本地 prefix cache。命中次数取决于"是否黏同一副本 + 前缀是否重复 + 本地是否装得下"。

### 9.6 选型逻辑的来源与边界

"要容量/多租户选 Mooncake，要 A3 低延迟选 Memcache"这句话**来自官方场景表**（[kv_cache_pool_ascend_store.md L17-L18](docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L17-L18)），不是从"对象 vs 地址"这个操作维度推出来的。两者是**两个正交层次**：

| 层次 | 内容 | 决定什么 |
|---|---|---|
| 数据平面 | `block_key` vs `gva` | **怎么搬**（§4） |
| 系统能力 | SSD、配额、HCCS、生态 | **装多少、多快、谁管** |

有倾向性、无必然性：对象语义天然带元数据服务 → 配额/淘汰/多租户顺手长出来，但每次访问多一次元数据往返；地址语义天然省掉这趟往返 → 往低延迟走，但 alloc/租约/可见性要自己管。两家都在往对方地盘扩（各有 SSD 层）。

**两处需要修正的说法**：

1. ❌"以逐层搬运为主选 Memcache"——Mooncake ≥ 0.3.14 也支持 layerwise，两者都能逐层搬，差别只有协议和版本门槛。
2. ❌"低延迟是 Memcache 独占"——Mooncake 在 Ascend 上也走 `AscendDirectTransport`（HCCS/RDMA）；Memcache 的差异点在 MemFabric 的 GVA/UB 一侧 + `device_sdma`。

更严谨的按"你缺什么"选：

| 你的约束 | 指向 | 依据 |
|---|---|---|
| 要容量（KV 装不下 DRAM） | Mooncake | 有成熟的 SSD Offload 与淘汰策略 |
| 要多租户隔离/配额 | Mooncake | `tenant_id` + 配额快照 + metrics |
| A3 超节点、追求 P 侧传输延迟 | Memcache | MemFabric + UB/`device_sdma` |
| 不想多引一套依赖 | Mooncake | 默认后端，生态最广 |
| P/D P2P 与池化同一套栈 | Mooncake | P/D connector 本身就是 Mooncake 的 |
| A3 上 MemCache 与 vLLM 分离部署 | Memcache | 只有 A3 支持该形态 |
