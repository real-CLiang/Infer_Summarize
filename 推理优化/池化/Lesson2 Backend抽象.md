# 02 · Lesson 2 知识点：Backend 抽象（入门版）

> 读法：本文每一节都是**先图/先比喻，再代码**。看不懂代码时只看图也能过关。
> 文件路径相对仓库根 `vllm-ascend`，行号于 2026-10-31 核对。
> 先修：`KV Cache池化学习路线.md`（L0）、`Lesson1 全局架构学习.md`（L1）。

---

## 0. 全景：一图看懂 Backend 抽象

> 这一节是全文的**地图**。后面每一节讲的都是这张图里的某一个框。读到一半不知道自己在哪时，回到这里。

### 0.1 结构框图

```
┌─ 调用方 ───────────────────────────────────────────────────────────────┐
│ EngineCore（纯 CPU，只调度）           Worker × TP rank（手握显存地址）   │
│   KVPoolScheduler                        KVPoolWorker                  │
│      │ create_scheduler_client()            │ Backend(parallel_config) │
└──────┼──────────────────────────────────────┼──────────────────────────┘
       ▼                                      ▼
┌─ 两个入口：同一层，两套权限 ────────────────────────────────────────────┐
│  轻量 client：只能查                    完整 backend：能查 + 能搬        │
│  exists / batch_is_readable             + register_buffer + put / get  │
└──────┬──────────────────────────────────────┬──────────────────────────┘
       └───────────────┬──────────────────────┘
                       ▼
┌─ 契约：Backend ABC（backend/base.py，175 行）──────────────────────────┐
│ ① 必实现 6    __init__ set_device register_buffer exists put(2D) get(2D)│
│ ② 默认实现 3  create_scheduler_client close batch_is_exist             │
│ ③ 会话组      put_start copy_put commit revoke                         │
│  （对象语义） get_start copy_get get_end validate_layerwise_support    │
│ ④ 空间组      alloc add_lease remove_lease write_finish                │
│  （地址语义）                                                          │
│ ⑤ 显式泄漏    requires_exists_before_put                               │
└───────────────┬───────────────────────────────────┬───────────────────┘
                ▼                                   ▼
┌─ 注册表 backend/__init__.py ─────────┐  ┌─ 实现 ──────────────────────┐
│ backend_map[name] = {                │─►│ MooncakeBackend   block_key │
│   name, path, layerwise_protocol }   │  │ MemcacheBackend   gva       │
│ LAYERWISE_DATA_PLANE = 寻址方式声明   │  │ YuanrongBackend   (未声明)  │
└──────────────────────────────────────┘  └──────────────┬─────────────┘
                                                          ▼
                                        Mooncake Master / Memcache MetaService
```

**读图方向：从上往下 = 一次调用从上层走到下层。** 横向只有一处分叉——"两个入口"，其余都是单向。

### 0.2 这张图在说什么（三句话）

1. **上层有两个入口**，因为两个进程手里握着的东西不同：scheduler 只有 CPU 和 request，worker 才有显存地址。**权限不同，所以客户端不同。**
2. **中间的契约只认两样东西**：一个字符串 key，一组 `(地址, 长度)`。6 个必实现是"最小公共集"，两组可选方法是"后端能力的声明"。
3. **底下换仓库只动两处**：注册表加一项 + 配置里写 `"backend": "mooncake" | "memcache"`。上层代码一行不改。

### 0.3 读图要点（6 条）

1. **两个入口 = 两套权限**：scheduler 侧从不 `register_buffer`，所以它"能查不能搬"——这是 L1 里 bulk 命中要绕 ZMQ 问 worker 的根本原因。
2. **6 个必实现 = 最小公共集**：`register_buffer` 是硬约束（DMA 只肯访问登记过的内存）；`put/get` 二维 = 外层 key、内层段。
3. **可选 14 个分两组，几乎不重叠**：会话组属对象语义（空间在仓库手里），空间组属地址语义（空间在你手里）。**这就是"两家 API 长得不一样"的答案**——不是风格，是物理模型不同。
4. **`layerwise_protocol` 是模块 + 普通函数，不是抽象方法**：前台在 CPU 进程里，绝不能 import 两家的 C++ 库。代价是没有编译期检查，报错会晚到。
5. **`LAYERWISE_DATA_PLANE` 是能力声明**：`block_key` / `gva` 由后端决定，不能运行期切换；换它 = 换 key 空间 = 老池子零命中。
6. **抽象盖不住的三类东西**：控制流不同 → 变开关（`requires_exists_before_put`）；性能不同 → 可选加速（`batch_is_exist`）；物理原语不同 → 漏到调用方（直调 `store.batch_copy(gva, addr, size, dir)`）。

### 0.4 图 ↔ 章节对照（定位用）

| 图里的位置 | 本文哪一节 | 一句话 |
|---|---|---|
| 两个入口 | §3.2、§8 | 前台轻、仓管重，三个"不"（不虚增容量/不建 NPU 上下文/不参加 DP 同步） |
| 必实现 6 个 | §3.1、§4 | `register_buffer` 是硬约束；`addrs` 为什么二维 |
| 会话组 7 个 | §5.1、§6 | 对象语义：开单 → 逐层写 → 提交 |
| 空间组 4 个 | §5.2、§6 | 地址语义：申请 → 逐层写 → 宣布完成 + 借条 |
| `requires_exists_before_put` | §9 | 抽象盖不住，只能变成开关 |
| `backend_map` / `LAYERWISE_DATA_PLANE` | §7 | 为什么是模块常量而非抽象方法 |
| 换后端 / 接新后端 | §10 | 整块 4 步；layerwise 再 +4 步 |

### 0.5 30 秒版（只想先有个印象，读这三句）

1. **后端（Backend）就是"仓库的存取接口"**：只认两样东西——一个字符串 key，和一组 `(地址, 长度)`。
2. **Mooncake 是"对象仓库"**：你把 key 给它，它自己找地方存、自己管什么时候能被别人看见。所以接口是"开单 → 分批写 → 提交"。
3. **Memcache 是"共享内存"**：它给你一段地址，写哪儿、什么时候算写完、读完怎么还，全要你自己管。所以接口是"申请 → 写 → 宣布写完 + 借条"。

后面所有细节，都是这三句话的展开。

---

## 1. 先建四个直觉（后面每节都挂在这四张图上）

### 图 1：两个进程，一个仓库

```
┌──────────── EngineCore 进程（前台 · 只调度，不碰显存）────────────┐
│  Scheduler ──► "这段前缀算过吗？能省几个 token？"                  │
│      │                                                          │
│      │ create_scheduler_client()      ← 只建"能查"的轻量客户端      │
│      ▼                                                          │
│  ┌──────────────────┐                                           │
│  │ 轻量客户端（只读）  │                                           │
│  └────────┬─────────┘                                           │
└───────────┼─────────────────────────────────────────────────────┘
            │ 只查账：这个 key 在不在、能不能直接读
            ▼
   ╔════════════════════════════════╗
   ║ 外部存储服务（总账）              ║
   ║ Mooncake Master                ║
   ║ Memcache MetaService           ║
   ║ 记的是：key → 数据在哪台机器      ║
   ╚═══════════════╤════════════════╝
                   ▲
                   │ 真搬 KV 字节（DMA，不经过 CPU）
┌──────────────────┼──────────────────────────────────────────────┐
│                  │                                              │
│  ┌───────────────┴────────┐                                     │
│  │ 完整客户端（能读也能写）   │  ← 默认行为就是长这样                 │
│  │ register_buffer 登记显存  │                                     │
│  │ put / get / 逐层搬       │                                     │
│  └───────────┬────────────┘                                     │
│              ▼                                                  │
│           NPU 显存（KV cache 住这儿）                              │
│                                                                 │
│  Worker 进程 × TP rank 数（仓管 · 手里才有显存地址）                 │
└─────────────────────────────────────────────────────────────────┘
```

**这张图要说的事**：前台和仓管连的是同一个仓库，但**前台只需要"查"，仓管才需要"搬"**。

### 图 2：后端眼里的世界

```
       vLLM 的世界                            后端的眼里
┌──────────────────────────┐          ┌────────────────────────┐
│ 请求、block、层、attention │  翻译成   │  "abc123" 这个 key      │
│ 命中、调度……              │ ───────► │  + [地址1, 地址2, ...]   │
└──────────────────────────┘          │  + [长度1, 长度2, ...]   │
                                       └────────────────────────┘
```

**这张图要说的事**：后端**完全不知道 vLLM 是什么**。它只知道"有个字符串，有些字节要放在这些地址上"。所有 vLLM 的语义（前缀、层、TP）都在到达后端之前被翻译成了 key + 地址。

### 图 3：同一个 key，在显存里是散的

```
【想象中】一个 block = 一整块连续内存
   0x1000_0000 ┌──────────────────────┐
               │ 这个 block 的全部 KV   │  20 MB 连着放
               └──────────────────────┘

【实际上】显存是按"层"分配的，一个 block 的 KV 散在各层
   层 0 的 K: 0x1000_0000 ┌────┐ 256 KB
   层 0 的 V: 0x1004_0000 ┌────┐ 256 KB
   层 1 的 K: 0x1008_0000 ┌────┐ 256 KB
        ...                 ...    共 44 层 × 2 = 88 段，互不相邻
   层43 的 V: 0x1F00_0000 ┌────┐ 256 KB
```

**这张图要说的事**：这就是 `put/get` 的地址要写成"二维"的唯一原因（第 4.2 节详讲）。

### 图 4：两版客户端

```
   默认版  Backend(parallel_config)          三家实际用的"轻量版"
   ─────────────────────────────            ────────────────────────────
   建一个"完整仓管"：                         建一个"只查台账的前台"：
   · 申请 1 GB 内存捐给池子                    · 不捐内存
   · 占一张 NPU 卡、建设备上下文                · 不占卡
   · 注册 SSD 目录                            · 不碰 SSD
   ·（Memcache）等一个同步信号 → 可能卡死        · 只保留"查"
```

**这张图要说的事**：同一份代码，通过**构造参数**区分前台和仓管（第 3.2 节详讲）。

---

## 2. 本讲要回答的两个问题

1. **为什么 Mooncake 和 Memcache 的接口长得不一样？**
2. **接一种新存储，要改哪一层、写哪些函数？**

---

## 3. 契约：这个"接口"一共要求什么

`Backend` 是个**抽象基类**（只写接口、不写实现的父类），文件在
[backend/base.py L73-L175](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py#L73-L175)。

它一共 23 个成员，分成三档：

### 3.1 六个"必须实现"（少一个，类就建不出来）

| 函数 | 人话 |
|---|---|
| `__init__` | 读自己的配置、连上仓库 |
| `set_device` | 绑定 NPU 设备（不然 DMA 不知道该往哪写） |
| `register_buffer` | 把显存地址"登记"给硬件（第 4.1 节） |
| `exists` | 问：这些 key 在不在？返回每 key 一个 0/1 |
| `put` | 批量写 |
| `get` | 批量读 |

### 3.2 三个"有默认实现"（不写也能用）

| 函数 | 默认行为 |
|---|---|
| `create_scheduler_client` | `return cls(parallel_config)`：**按完整流程再建一个** |
| `close` | 什么都不做 |
| `batch_is_exist` | 直接转给 `exists`（语义对，但慢） |

#### 3.2.1 重点：`create_scheduler_client` 的默认值到底在做什么

**它是个工厂函数**（`@classmethod`），作用是"给前台建客户端"。默认实现只有一行：

```python
@classmethod
def create_scheduler_client(cls, parallel_config):
    return cls(parallel_config)      # 什么都不传，全用默认值
```

**问题**：默认值就是"完整仓管"。三家后端都把它改轻了，每家只多传一个开关：

| 后端 | 重写后 | 开关的作用 |
|---|---|---|
| Mooncake | `cls(pc, contribute_memory=False)` | 别把本机内存捐给池子、别注册 SSD |
| Memcache | `cls(pc, device_id=..., init_bm=False)` | 别绑设备（不建 NPU 上下文）、别等 DP 同步 |
| Yuanrong | `cls(pc, enable_data_plane=False)` | 别开数据通道，只查账 |

开关在 `__init__` 里怎么生效（挑两个看）：

```python
# mooncake_backend.py
global_segment_size = self.config.global_segment_size if self._contribute_memory else 0
# memcache_backend.py
if self._init_bm:
    self.set_device()                 # 不加这句，前台就不会建 NPU 上下文
if self._init_bm and self._dp_init_barrier:
    torch.distributed.barrier(...)    # 不加这句，前台可能卡在一个等不到的同步上
```

**而仓管那边根本不用这个工厂**，它直接 `Backend(parallel_config, **kwargs)`，三个开关全用默认的 `True`（[pool_worker.py L456-L468](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L456-L468)）。所以整件事可以压缩成一行：

```
仓管 : Backend(pc)                               → 开关全 True  → 重
前台 : Backend(pc, contribute_memory=False, ...)  → 开关全 False → 轻
```

这条约定**靠测试守，不靠类型**：[test_backend.py L81-L104](tests/ut/distributed/ascend_store/test_backend.py#L81-L104) 断言的是 `npu.set_device.assert_not_called()`、`store.init.assert_called_once_with(2, init_bm=False)`。

> 这个默认实现当前**没有任何生产后端在用**（三家全覆写了）。它存在只是为了"新后端不写这一行也能跑"。

### 3.3 十四个"可选"（默认全部报 `NotImplementedError`）

分成两组，**两组几乎不重叠**——这就是第一个问题的答案：

| 组 | 方法 | 谁会实现 |
|---|---|---|
| **会话组**（对象仓库用） | `put_start` / `copy_put` / `commit` / `revoke` / `get_start` / `copy_get` / `get_end` | Mooncake |
| **空间组**（共享内存用） | `alloc` / `add_lease` / `remove_lease` / `write_finish` | Memcache |

### 3.4 三家后端的能力对照

| | Mooncake | Memcache | Yuanrong |
|---|---|---|---|
| 6 个必须项 | ✅ | ✅ | ✅ |
| 会话组 | ✅ | ❌ | ❌ |
| 空间组 | ❌ | ✅ | ❌ |
| `batch_is_readable`（能不能直接读） | ✅ | ✅ | ❌ |
| `requires_exists_before_put` | True | True | **False** |
| 数据面类型 | `block_key` | `gva` | 没声明 |

> 一句话：**不是风格不同，是两套方法集根本对不上。** 为什么对不上，看第 5 节。

---

## 4. 三个"必须实现"的背后，各有一个物理原因

### 4.1 `register_buffer`：为什么必须先"登记"

**图：为什么不能直接写**

```
   不登记：                          登记过：
   网卡 ──X──► 显存某地址             网卡 ──✓──► 显存某地址
   （硬件不知道这块内存归谁，          （登记时告诉了驱动：这块内存
     直接拒绝 / 静默写错地方）           可以被远程写，也叫 pin 住）
```

跨机搬 KV 走的是 DMA（HIXL / RDMA / UB / SDMA），**不经过 CPU**。这类硬件只肯访问**登记过**的内存。所以 worker 启动时会调一次：

```python
# pool_worker.py L1346-L1353
if self.use_layerwise_transfer:
    self.m_store.ensure_initialized()
self.m_store.register_buffer(ptrs, lengths)      # ① 所有后端都做
if self.use_block_key_layerwise:
    self.m_store.validate_layerwise_support()    # ② 只有 Mooncake 要查客户端版本
self._start_kv_transfer_threads()
```

三家登记到的地方不同：Mooncake 走 `store.register_buffer` 或全局传输引擎；Memcache 走 `store.register_buffer`，而且**仓库还没初始化时先把地址暂存起来**，等初始化完再补登记；Yuanrong 只在特定硬件组合下需要。

> 记法：**没登记 = 一次都搬不了。** 这也是"精度异常但不报错"的头号来源。

### 4.2 `put/get` 的地址为什么是二维的（本讲最容易卡的一节）

#### 4.2.1 一句话

因为**一个 key 对应的那块 KV，在显存里不是一整块，而是散在很多块里**。

#### 4.2.2 看图

```
【想象中】一个 block = 一整块连续内存
   0x1000_0000 ┌──────────────────────────┐
               │  这个 block 的全部 KV      │  22 MB 连着放
               └──────────────────────────┘
   那 put 写成这样就够了（一维）：
   put(keys=["A"], addrs=[0x1000_0000], sizes=[22MB])
```

```
【实际上】显存是按"层"分配的，一个 block 的 KV 散在各层里
   层 0 的 K: 0x1000_0000 ┌──────┐ 256 KB
   层 0 的 V: 0x1004_0000 ┌──────┐ 256 KB
   层 1 的 K: 0x1008_0000 ┌──────┐ 256 KB
   层 1 的 V: 0x100C_0000 ┌──────┐ 256 KB
        ...                  ...      共 44 层 × 2 = 88 段
   层43 的 V: 0x1F00_0000 ┌──────┐ 256 KB
              这些地址互不相邻！
```

所以真正要传给后端的是：**"A 这个 key，由这 88 段的地址和长度组成"**。

#### 4.2.3 于是签名的两层意思

```python
put(keys=["A", "B"],                     # 外层：这次要写 2 个 key
    addrs=[[88 个地址], [88 个地址]],      # 外层跟 keys 对齐；内层是这个 key 的 88 段
    sizes=[[88 个长度], [88 个长度]])
```

- 外层长度 = 有几个 key；
- 内层长度 = 这个 key 拆成了几段。

"二维"说白了就是**一串小串**。`addrs[i]` 和 `sizes[i]` 配在一起，才是第 i 个 key 的完整清单。

#### 4.2.4 代码里就是这个循环

`metadata.py` 的 `prepare_value()`（[L488-L528](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L488-L528)）——注意 `addr_list` 是个**列表**：

```python
addr_list = []
size_list = []
for index, base_addr in enumerate(group_addrs):   # 遍历这个 block 的每一段（每层每个张量）
    ...
    addr = base_addr + block_id * block_stride + byte_start
    addr_list.append(addr)                       # 一段一个地址 → 攒成内层
    size_list.append(byte_end - byte_start)
return addr_list, size_list, block_id
```

调用方（[kv_transfer.py L1414-L1467](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1414-L1467)）是"每个 key 收一份小清单"：

```python
for index, start in enumerate(starts):
    addr, size, _ = self._prepare_value(...)   # addr 已经是"88 个地址"的小列表
    addrs.append(addr)                         # 外层：一个 key 一份
    sizes.append(size)
...
put_result = self.m_store.put(keys, addrs, sizes)
```

#### 4.2.5 三个会卡住的疑问

**Q1：为什么不能拍平成一维？**
`addrs=[0x1000, 0x1004, ..., 0x1F00]` 这样一长串，后端**没法知道哪几段属于 key A、哪几段属于 key B**。你要么再传一个"每 key 几段"的分界数组——那等于把二维换个名字加回来；要么保证每个 key 段数一样——可实际不一样（尾块是半块、TP 不对齐时切得更碎）。

**Q2：为什么 `sizes` 也是二维？**
因为每段长度不一定相同。整块时每段都是 256 KB；但尾块、按 head 切片的场景长度会变。所以"长度"必须跟着"地址"一段一段走。

**Q3：为什么不是三维（层 × K/V × 段）？**
因为后端**不关心**这些段是"层"还是"K/V"。它只要"地址 + 长度"。层的语义在 `prepare_value` 里已经被拍平进内层列表了——对后端来说，88 段就是 88 段，一视同仁。

> **记法：外层 = 几个 key，内层 = 这个 key 散成了几段。**

### 4.3 `exists` 为什么只允许 0 和 1

```python
if all(state == 1 for state in 本 block 的状态):  命中
elif any(state == 0 for state in 本 block 的状态): 断在这里
else: raise RuntimeError(...)                     # 负数/其他 = 后端报错
```

[pool_scheduler.py L333-L356](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L333-L356)

有意思的是**同一个"问在不在"，两条路径的失败策略正好相反**：

| 谁在问 | 用途 | 遇到"后端报错"怎么办 |
|---|---|---|
| 命中查询 | 决定省多少 prefill | **报错停下**（宁可不服务，也不假装 miss） |
| 保存前过滤 | 决定写不写 | **当成"没有"直接写**（多写浪费带宽，少写会丢块） |

见 [pool_worker.py L2445-L2488](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2445-L2488)。读路径要正确性，写路径要可用性。

---

## 5. 两种仓库，两套接口

### 5.1 对象仓库（Mooncake）

**图：**

```
   你：把 "abc123" 这个 key 存进去，大小 22 MB
   仓库：好，我自己找地方放。放完之前别人看不见，放完（commit）才可见
   你：那我想分批写呢？
   仓库：开个"写作会话"（put_start），按对象内偏移一段段写，最后 commit
```

三个后果：

1. **空间不是你的**——你拿不到地址，写的时候只能说"写在对象的第几字节"（对象内偏移）；
2. **可见性是原子的**——没 commit 的对象，谁都读不到；
3. **commit 之后不能覆盖**（Mooncake 会报 `MMC_UNMATCHED_KEY`）——所以写之前必须先 `exists` 查一遍，别去动已经提交的对象。半成品怎么办？`revoke` 扔掉。

代码路径：`put_start = store.batch_put_session_start`、`copy_put = store.batch_put_from_multi_buffer_ranges`、`commit = store.batch_put_session_end`、`revoke = store.batch_put_session_revoke`（[mooncake_backend.py L378-L435](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L378-L435)）。

### 5.2 共享内存（Memcache / GVA）

**图：**

```
   你：我要存 "abc123"，22 MB
   仓库：给你这段地址（GVA = 0x7f00_0000），你自己写
   你：写完了
   仓库：好，我记一笔"已写完"（write_finish），别人现在能读了
   读者：我要读 → 先打借条（add_lease），读完还（remove_lease）
```

- **地址是你的**：`alloc` 会返回真实地址，写的时候直接给"目标地址"，**连 key 都不用传**；
- **写完 ≠ 可见**：必须 `write_finish` 宣布一下；
- **读之前要打借条**：`add_lease`（5 分钟有效），否则可能读到一半被回收；
- `alloc` 是**可重复的**：同一个 key 再申请返回同一个地址，不会报错（但重复写就白写了）。

代码：[memcache_backend.py L443-L459](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py#L443-L459)。

### 5.3 一张表看完两套接口

| 阶段 | 对象仓库（Mooncake） | 共享内存（Memcache） |
|---|---|---|
| 拿空间 | `put_start(keys, sizes)`：只登记意图，**不给地址** | `alloc(keys, sizes)`：**给地址** |
| 写 | `copy_put(keys, buffers, sizes, **对象内偏移**)` | `copy(**地址数组**, 本地地址, 大小, 方向)`，key 不参与 |
| 发布 | `commit(keys)` | `write_finish(keys)` |
| 放弃 | `revoke(keys)` | 没有对称操作（没发布的会被淘汰） |
| 读前 | `get_start(keys)` | `add_lease(keys, 5 分钟)` |
| 读 | `copy_get(..., 对象内偏移)` | `copy(..., 反方向)` |
| 读后 | `get_end(keys)` | `remove_lease(keys)` |
| 能不能读 | `exists`（没 commit 本来就看不见） | `get_key_info`：有大小且地址非 0 |

**一句话**：对象仓库**把地址藏起来**，所以接口以 key 为中心；共享内存**把地址交给你**，所以接口以地址为中心。

### 5.4 一个只在 Memcache 出现的坑：`exists` 为真，但读不到

Memcache 是**分级存储**：热数据在内存，冷数据落到 SSD。一个 key 可能存在，但数据在 SSD 上——**没有可以直接 DMA 的地址**。

```
   exists = 1（账上有）
        │
        ├─ 数据在 DRAM → gva 有值 → 能直接读
        └─ 数据在 SSD  → gva = 0  → 读之前要先"搬回内存"
```

代码里的两个应对：

1. 命中查询不用 `exists`，用 `batch_is_readable`（查元数据，[memcache_backend.py L418-L441](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py#L418-L441)）；
2. 真读的时候先 **rewarm**：临时申请一块 NPU 内存 → 登记 → 走一次完整读把 SSD 数据搬到 DRAM → 释放（[L326-L374](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py#L326-L374)）。

Mooncake 没有这个问题：没 commit 的对象根本看不见，所以 `exists` 就等于"能读"。

> 这也是抽象层**真正挣到钱**的地方：前台只问一个问题——"这块能直接读吗"，两家的不同语义在各自后端里被翻译掉。

---

## 6. layerwise：一层一层搬的时候，三套不同的做法

### 6.1 先看一个开关

三个文件里都有同一段判断（connector / scheduler / worker）：

```python
plane = 后端模块里的 LAYERWISE_DATA_PLANE        # "block_key" | "gva" | 没有
用对象仓库那套 = use_layerwise and plane == "block_key"
用共享内存那套 = use_layerwise and plane == "gva"
```

**注意**：如果后端没声明这个常量，两个布尔都是假，但**不会有任何报错**——它会悄悄走第三套历史路径（下一节）。

### 6.2 三套并存的现实（修正 L1 的说法）

L1 说的是"两种数据面"，实际是三套：

| | `block_key`（Mooncake） | `gva`（Memcache） | 没声明（历史路径） |
|---|---|---|---|
| 一个对象是什么 | 1 个 block × 1 个 head 的**整层栈** | 1 个 block 的整层栈（一段地址） | **1 个 block 的 1 层** |
| key 长相 | `模型@{hash}@{head}` | `模型@[group]@{hash}@{head}` | 多带一个 `@layer_id:` |
| 写 | 开单 + 逐层写 + 提交 | 申请 + 逐层写 + 宣布完成 | 每层直接 `put` |
| 读 | 开读单 + 逐层读 + 关单 | 打借条 + 逐层读 + 还 | 每层直接 `get` |
| 用哪个线程 | `LayerSendingThread`（key-major 分支） | `LayerSendingThread`（GVA 分支） | `KeyLayerSendingThread` |

> **对 L1 的修正**：L1 把 `KVCacheStoreKeyLayerSendingThread` 记成"Mooncake 的 block_key 保存线程"。现在 Mooncake 走的是 `LayerSendingThread` 的 key-major 分支（`copy_put + commit`）；`KeyLayer*` 是按层一个对象的**历史路径**。判断依据只有一条：`task.use_key_major_ranges`（[kv_transfer.py L2161](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L2161)、[L2455](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L2455)）。

### 6.3 一次保存的时间线（两边其实很像）

```
请求 R，要存 6 个 block，每块 20 MB，44 层

[对象仓库 / Mooncake]                        [共享内存 / Memcache]
开单：put_start(6 个 key, [20MB]*6)          申请：alloc(6 个 key, [20MB]*6) → 6 个地址
        │                                            │
第 0 层：等显存写完 → copy_put(偏移 0)         第 0 层：等显存写完 → copy(地址+0)
第 1 层：等显存写完 → copy_put(偏移 1 层)      第 1 层：等显存写完 → copy(地址+1 层)
   ...（发送线程最多领先 8 层）                    ...
第43 层：copy_put(最后一层)                    第43 层：copy(最后一层)
        │                                            │
提交：commit(6 个 key) → 此刻才可见            宣布：write_finish(6 个 key) → 此刻才可读

出错：revoke(失败的 key)                       出错：停止本步保存（不删别人的 key）
```

关键点：**两边都是"逐层写、最后一次性发布"**。区别只在最后那个动作叫 `commit` 还是 `write_finish`。

### 6.4 为什么会话还需要一本"账"

一个读会话是**跨多个计算步**活着的，而且**多个请求可能读同一个 key**（同一段前缀）。所以需要记：

- 谁开了这个 key 的会话；
- 谁还在读 → 引用计数；
- 只有**最后一个**读者退出，才能真正"关单"（`get_end`），否则会把别人的读通道关掉。

这本账就是 `LayerwiseSessionTracker`（[session_tracker.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/session_tracker.py)）。

> **bulk 的 `put` 是一次函数调用；layerwise 的 `commit/get_end` 是一段有状态的过程。** 14 个可选方法就是这么来的。

---

## 7. 为什么"数据面类型"是个模块常量，而不是一个函数

后端注册表长这样（[backend/__init__.py L22-L37](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/__init__.py#L22-L37)）：

```python
backend_map = {
  "mooncake": {"name": "MooncakeBackend", "path": "...mooncake_backend",
               "layerwise_protocol": "...mooncake_layerwise"},
  "memcache": {"name": "MemcacheBackend", "path": "...memcache_backend",
               "layerwise_protocol": "...memcache_backend"},
  "yuanrong": {"name": "YuanrongBackend", "path": "...yuanrong_backend"},
}
```

`layerwise_protocol` 指向的是**一个模块**（不是类），这个模块要提供一组**普通函数**：

| 模块里的东西 | 干什么 |
|---|---|
| `LAYERWISE_DATA_PLANE` | 声明自己属于哪一套（`block_key` / `gva`） |
| `bind_layerwise_keys(...)` | 按自己的格式拼 key |
| `make_hit_check_keys(...)` | 一个 block 要查哪些 key（所有 rank、所有 PP 段） |
| `extract_layout_config(...)` | 要不要走共享内存的物理复用 |
| `validate_topology` / `validate_runtime` | 启动时拒绝不支持的组合 |

**为什么不用类的抽象方法**：前台运行在 EngineCore（纯 CPU 进程），它需要 key 的格式和校验规则，但**绝对不能 import Mooncake / Memcache 的 C++ 库**（一 import 就可能拉起 NPU 上下文）。模块级函数是"只共享格式、不共享句柄"的最小办法。

**代价**：没有编译器检查你少写了哪个函数，报错会晚到（`AttributeError`，甚至第一次请求才炸）。

---

## 8. 一个仓库，两个入口

```
                  ┌──────────────────────────────┐
   前台 ────────► │ create_scheduler_client()    │→ 只读：在不在、能不能读
                  ├──────────────────────────────┤
   仓管 ────────► │ Backend(...)                 │→ 登记显存 + 读写 + 全部 layerwise
                  └──────────────────────────────┘
```

三家前台客户端的取舍理由（都有测试守着）：

1. **不虚增容量**：前台要是也注册 SSD，仓库会以为"又来一个捐硬盘的"；
2. **不建 NPU 上下文**：前台是 CPU 进程，白占一张卡；
3. **不参加 DP 同步**：前台跟着等，会卡在一个永远不来的信号上。

这也回答了 L1 那个遗留问题：**"能查"和"能搬"是两套权限**。前台手里的客户端**从来没有登记过任何显存地址**，所以普通 bulk 的命中还得打内部电话去问仓管。

---

## 9. 抽象盖不住的地方（原"泄漏清单"）

| 盖不住的东西 | 长成了什么 | 为什么盖不住 |
|---|---|---|
| "重复写同一个 key"的语义不同 | 参数 `requires_exists_before_put` | 它决定**调用方**要不要多查一轮，`put` 内部决定不了 |
| 批量查询更快 | 可选函数 `batch_is_exist` | 语义一样，性能差一个数量级 |
| 共享内存的数据面 | 直接调 `store.batch_copy(地址数组, 本地地址, 大小, 方向)` | 抽象里根本没有"地址 + 方向"这个动作 |
| 特定协议的私有函数 | `unregister_buffer` / `init_store` / `put_from` | 只在某些硬件协议下有意义 |
| 后端返回值的形状 | `BatchResultShapeError` + 长度校验 | C++ 接口可能少返回一个元素，只能运行期查 |

**判断规律**：语义一致但**代价/控制流不同**的 → 变成开关；**物理原语不同**的（对象 vs 地址）→ 一定漏到调用方。

---

## 10. 接第四种存储（本地 NVMe 为例）

### 10.1 只做整块读写（4 步）

1. 继承 `Backend`，实现 6 个必须项；
2. 定 `requires_exists_before_put`（三连问：重复写会报错吗？会破坏正在读的数据吗？重复写的带宽贵吗？任一为"是" → `True`）；
3. 在 `backend_map` 注册名字和路径；
4. 配置里写 `"backend": "你的名字"`。

### 10.2 再加 layerwise（对象语义，4 步）

5. 写一个协议模块，声明 `LAYERWISE_DATA_PLANE = "block_key"` + 三个函数；
6. 注册表里补 `layerwise_protocol`；
7. 实现会话组七个函数 + `validate_layerwise_support`；
8. 加一个"key 字符串格式"的快照测试（key 是跨机器、跨版本的约定，改一个字符老池子就全 miss）。

### 10.3 再加 layerwise（地址语义）

把第 5~7 步换成 `"gva"` + 空间组五个函数，并且你的库要能提供 `batch_copy(目标地址, 本地地址, 大小, 方向)`。

---

## 11. 坑表

| 现象 | 真实原因 | 先看哪里 |
|---|---|---|
| 开了 layerwise，行为却像"每层一个对象" | 后端没声明数据面 → 悄悄走历史路径 | 实际是哪个线程类被创建 |
| `exists` 命中，`get` 却失败 | 数据在 SSD，没有可读地址 | `batch_is_readable`、`get_key_info` 的介质类型 |
| 换后端后命中率瞬间 0 | 换了数据面就换了 key 格式，两个池子互不认识 | 两边的 `make_*_key` |
| 日志里 `MMC_UNMATCHED_KEY` | 试图覆盖 Mooncake 已提交的对象 | 写之前有没有先 `exists` 过滤 |
| 精度偶发异常，没有任何报错 | buffer 没登记 / 会话没 commit 就去读 | `register_kv_caches` 和 commit 时序 |
| 后端返回 `None` 被当成成功 | 返回值有 bool / 列表 / None 三种形状 | `put`/`get` 的返回值处理 |

---

## 12. 自检题

### 基础 3 题（答不出就回看对应节）

1. 用一句话说清：为什么前台客户端要"轻"，仓管要"重"？（第 3.2 节）
2. 一个 44 层、每层 K/V 两个张量的模型，一个 block 的 KV 在显存里大概散成几段？为什么 `put` 的 `addrs` 必须是二维？（第 4.2 节）
3. Memcache 的 `exists` 返回 1，为什么还可能读不到？（第 5.4 节）

### 进阶 2 题

4. 用"谁拥有空间、谁拥有可见性"解释：为什么一个是 `put_start → commit`，另一个是 `alloc → write_finish`？
5. 接一个本地 NVMe 后端（整块 + layerwise，对象语义），列出必须实现的函数、注册表改动、`requires_exists_before_put` 的取值和理由。

### 加练 2 题（能答说明真读懂了）

6. 为什么 `LAYERWISE_DATA_PLANE` 是**模块常量 + 普通函数**，而不是 `Backend` 的一个抽象方法？
7. 为什么命中查询遇到 `exists` 返回 -1 要报错，而保存前过滤遇到 -1 却当"没有"直接写？

---

## 13. 通往 L3

L2 只讲了"key 是个字符串"，没讲**字符串里该有什么**。第 6.2 节那张表已经暴露三套格式：

- `block_key`：带 PP 段号、可选命名空间、可选 hybrid 布局；
- `gva`：带 group 号、带 PP 号，单 group 时退回老格式；
- 历史路径：带 `@layer_id`、`@dcp`、`@cache_role`、`@cache_family`。

**同一份 worker 代码要同时支持三种拼法**，靠的就是 `bind_layerwise_keys` 返回的 `LayerwiseKeyBuilder`。

L3 回答：这些字段各补哪个缺失维度、TP 变了怎么重写 key、为什么命中点必须对齐到"传输粒度"。

---

## 14. 待验证点

1. 第 5.4 节的 SSD rewarm 只读过代码，**未在真机验证**（需要 Memcache + SSD）。
2. 第 6.2 节"没声明数据面 → 走历史路径"是**代码推断**（线程选择逻辑 + 全程无告警），文档里没写 Yuanrong 是否允许 `use_layerwise=True`。
3. `batch_copy` 直调库函数，目前只有 Memcache 的地址平面用到；未来是否补进抽象层，未知。
4. 未运行 `pytest tests/ut/distributed/ascend_store/test_backend.py -q`（缺 vllm/mooncake 环境），本文结论均来自源码阅读。
