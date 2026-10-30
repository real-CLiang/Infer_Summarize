# KV Cache 池化 与 vLLM 原生 Prefix Caching 对比与关联

> 面向昇腾（MindIE/灵衢）、vLLM-Ascend 推理优化场景的技术梳理。

***

## 1. 一句话核心区分

| 概念 | 一句话定位 |
| --- | --- |
| **vLLM 原生 Prefix Caching（APC）** | 单实例、单 GPU 内部的「**块哈希复用策略**」，解决"同一个 vLLM 进程里相同前缀不用重算 Prefill" |
| **KV Cache 池化（KV Pooling / Unified KV Pool）** | 跨卡、跨实例、跨节点的「**全局共享存储底座**」，Prefix Cache 只是跑在这个底座上的上层复用逻辑 |

> 昇腾 MindIE 官方文档明确：**KV Cache 池化必须同时打开 Prefix Cache**。池化是承载前缀缓存的分布式池子，不是替代前缀缓存。

***

## 2. vLLM 原生 Automatic Prefix Caching（APC）

### 2.1 作用范围

**局限在同一个 vLLM Worker / 同一个 GPU / 同一个 TP 组内部。**
不同 GPU、不同 vLLM 实例之间相互隔离，各自维护自己的 block 哈希表，互不认识对方的 KV 块。

### 2.2 工作原理

1. 基于 PagedAttention，KV 切成固定大小 block（如 16 token/块）；
2. 对每个 block 做**链式哈希**：`hash = hash(父块hash + 当前块token + 附加信息)`；
3. 维护**全局哈希表**：hash → 本卡内物理 block；
4. **命中**：多个请求的 block table 指向同一块显存 block，`ref_cnt` 引用计数 +1，写时复制（COW）；
5. **淘汰**：显存满、块被回收覆盖时，对应 hash 条目失效。

### 2.3 短板

- 缓存容量 = **本卡 HBM 中分给 KV 的那部分**；
- 多副本负载均衡、请求漂移到别的卡 → 缓存直接 miss，重新 Prefill；
- 只能停留在本地 HBM，**没有自动分级卸载**（HBM → DDR → SSD）、跨节点 RDMA 拉取。

### 2.4 定位

**复用策略 / 查找索引层**，本身不提供跨设备共享存储。

***

## 3. KV Cache 池化（KV Pooling）

### 3.1 作用范围

**全局分布式池子：多 NPU、多 Worker、多节点、HBM + DDR + SSD 多级介质。**
是独立的共享存储层，与 vLLM 引擎解耦。

### 3.2 工作原理

1. 所有参与池化的卡 / 节点共同贡献内存，拼成一个**统一 KV 资源池**；
2. 上层**依然用 Prefix Cache 那套「token 前缀哈希 / radix 树」做命中判断**；
3. 命中后：KV 块**不在本地就走 RDMA / 高速互联远程拉取**，而不是本地查不到就 miss；
4. 自带**全局 LRU 淘汰、冷热分级**：热块留在 HBM，冷块自动下放到主机内存 / SSD；
5. 支持 **PD 分离**：Prefill 集群算完 KV 丢进全局池，Decode 集群直接读。

### 3.3 定位

**底层存储底座**，负责 KV 块的跨设备寻址、传输、分级、持久化；前缀缓存是它的"检索规则"。

***

## 4. 逐项对比表

| 维度 | vLLM 原生 Prefix Caching | KV 池化（KV Pool） |
| --- | --- | --- |
| 作用域 | 单 Worker、单卡内部 | 跨卡、跨实例、跨节点集群 |
| 核心能力 | 识别相同 token 前缀，本地显存块复用 | 构建全局 KV 资源池，跨介质 / 跨设备共享 + 卸载 |
| 依赖 | 依赖 PagedAttention 本地 block 管理 | **依赖 Prefix Cache 做前缀匹配，池化是承载层** |
| 缓存上限 | 单卡 HBM 容量 | 集群聚合 HBM + DDR + SSD，容量大很多 |
| 远程访问 | 不支持，跨卡直接 miss | 支持 RDMA 远程取 KV 块 |
| 冷热分级卸载 | 无（仅本地 HBM） | 有（HBM ↔ DDR ↔ SSD 自动迁移） |
| 典型场景 | 单机单卡、简单多卡 TP，请求不漂移 | 大集群、Agent 长会话、PD 分离、高并发共享 system prompt |

***

## 5. 两者的关联：分层架构

在昇腾 vLLM-Ascend 场景中，二者是**上下两层、协同工作**的关系：

```
用户请求
  → 提取 token 前缀哈希（Prefix Cache 逻辑）
  → 先查本地 HBM
  → 本地没命中 → 去【全局 KV 池】查（灵衢互联 / HCCL）
      → 池里有：RDMA 拉回 KV 块，直接复用，跳过 Prefill
      → 池里没有：跑 Prefill，算完 KV 写入全局池，供所有人复用
```

- **Prefix Cache 回答**：这段前缀是不是已经算过？
- **KV 池回答**：算过的 KV 块存在哪里、能不能跨卡拿过来用？

> 简言之：**Prefix Cache 是"查找规则"，KV Pooling 是"存储底座"。前者决定命中与否，后者决定命中后能不能跨设备拿到。**

***

## 6. 一个直观例子

- **只开原生 Prefix Cache**：8 卡 TP，8 个卡各自独立缓存；一个请求路由到卡 0 命中，切到卡 1 就 miss，重算；
- **开 KV 池 + Prefix Cache**：8 卡组成一个池；不管请求调度到哪一张卡，只要前缀在池里，都能拉取复用，**缓存命中率不再受调度路由影响**。

***

## 7. 易混淆的坑：PagedAttention 的 Block Pool ≠ KV 池化

| 名称 | 是什么 | 是否跨卡 / 做前缀共享 |
| --- | --- | --- |
| PagedAttention 本地 block pool | 本卡内**空闲显存块的分配池** | 否，仅本卡分配 |
| KV Cache 池化（灵衢 / UCM / Mooncake） | **跨设备、可远程访问的 KV 缓存共享池** | 是，独立组件 |

> PagedAttention 里那个"free block pool"只是本卡内存分配机制，跟我们要说的 KV Cache 池化（跨设备共享池）不是一回事。

***

## 8. 附带澄清：第二种"KV 池化"（同名混淆）

还有一类**算法层面的 KV 向量池化**（对连续一段 token 的 K/V 做 mean/max pooling 压缩 KV 数量），用途是**超长上下文压缩缓存、减少显存**，代价是信息丢失、注意力精度下降。

它与本文讨论的\*\*资源池化（共享复用、无损）\*\*是两码事，仅名词相近，需注意区分。

***

## 9. 一句话总结

> **vLLM 原生 Prefix Caching 是"单机内如何识别并复用相同前缀"的查找策略；KV Cache 池化是"跨设备如何统一存放、远程拉取、分级卸载 KV 块"的存储底座。池化必须建立在 Prefix Cache 的前缀匹配之上，前者是上层逻辑，后者是下层承载。**