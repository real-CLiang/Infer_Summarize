# 03 · Lesson 3 知识点：Key 与命中判定

> 读法：先读 §0 全景（一张图 + 六条要点），再进正文。每节仍是"先图/先比喻，再代码"。
> 文件路径相对仓库根 `vllm-ascend`，行号于 2026-10-31 核对。
> 先修：`KV Cache池化学习路线.md`（L0）、`Lesson1 全局架构学习.md`（L1）、`Lesson2 Backend抽象.md`（L2）。

---

## 0. 全景：Key 与命中判定

> 本讲只回答两个问题：**key 里该有什么**、**怎么用它判断"能省多少 token"**。

### 0.1 结构框图

```
┌── 半边一：指纹（vLLM 送的，池化不改）───────────────────────────────┐
│  block_hashes:  h0 ──► h1 ──► h2 ──► ... ──► hN                     │
│  h_k = H( h_{k-1} , tokens[k*B : (k+1)*B] )                         │
│  → 每个 h_k 编码的是"从第 0 个 token 到本块末尾的整条前缀"            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼  半边二：坐标（KeyMetadata，池化自己补）
┌──────────────────────────────────────────────────────────────────────┐
│ model_name @ dcp_rank @ head_or_tp_rank @ pp_rank                    │
│            @ kv_cache_group_id @ cache_role @ cache_family           │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼  PoolKey.to_string()
        "qwen3@dcp:0@head_or_tp_rank:3@pp_rank:0@group:0@cache_role:kv
         @cache_family:default@3f9a…c1"      ← 全局唯一的对象名
                                │
                ┌───────────────┴────────────────┐
                ▼                                ▼
┌── 命中判定：exists + 对齐 ────────┐  ┌── 传输：put/get（L2 已讲）──────┐
│ ① 从 block 0 起逐块问             │  │ addrs[key][segment]（二维）      │
│ ② 第一个"非全 1"的块就停          │  │ 与 key 一一对应                  │
│ ③ 命中点必须落在 granularity 上   │  └──────────────────────────────────┘
│ ④ 多 group 取 min                 │
└──────────────┬───────────────────┘
               ▼
      hit_tokens = 连续块数 × block_size
               ▼
      再被三刀压（L1 §2.6）→ 真正的 num_external_hit_tokens
```

**读图方向**：左边是"内容"，中间是"坐标"，右边是"用法"。**key = 指纹 × 坐标，缺一个就是静默错数据。**

### 0.2 这三块分别在说什么（三句话）

1. **指纹**解决"这段 token 是不是同一段"，**坐标**解决"这份 KV 是不是同一个物理分片"。两者相乘才是池子里唯一确定的那份字节。
2. **坐标有七个维度**，每一个都对应一种"同一段 token 却字节不同"的并行/模型形态（TP、PP、DCP、group、role、family、layer）。
3. **命中判定不是查表，是"沿链逐块比对 + 对齐到传输粒度"**——它的输出是"从第 0 个 token 起连续可复用多长"。

### 0.3 读图要点（6 条）

1. **链式 hash 让"前缀复用"变成"从 h0 起连续存在"**——不需要任何"是不是前缀"的额外判断（L0 §5.3 推论一）。
2. **坐标不是元数据，是身份**：少一个维度就是静默错数据，这类 bug 不报错、只算错。
3. **粒度不是一个而是三个**：`hash_block_size`（算 hash 用）、`group_block_size`（每组自己的块大小）、`cache_transfer_granularity`（三者 + lcm 的传输粒度）。
4. **总开关是 `granularity`**：命中点必须落在它上面，否则传输会跨页、跨 head 切片。
5. **多 group 取 min**：任何一个 group 缺，都得从那里重算（`min(hits)`）。
6. **TP mismatch 不是"做格式转换"，而是"把坐标空间统一到 max(TP)"**：拆 sub-key、写 effective rank，让 P/D 两侧看到同一套 key。

### 0.4 图 ↔ 章节对照

| 图里的位置 | 本文哪一节 | 一句话 |
|---|---|---|
| 前置：TP / 权重 / KV | §0.5 | 不熟并行推理就先读这里 |
| 坐标为什么必须存在 | §2.1~§2.4 | 指纹并行无关 → 覆盖 / 四卡读到同一份 |
| group 是什么、`group_id` 从哪来 | §3.0、§1.4 | 按组存坐标模板；`group_id` 是循环下标 |
| 指纹（链式 hash） | §1.1 | 为什么 `get_block_hashes` 能直接取"最后一个细粒度 hash" |
| 坐标七维 | §1.2、§2 | 每个字段补哪个缺失维度、缺了会怎样 |
| `PoolKey.to_string()` | §1.3 | 三套拼法：完整 key / LayerPoolKey / 后端协议模块 |
| 三个粒度 | §3 | `hash_block_size` vs `group_block_size` vs `granularity` |
| 命中判定 | §4 | `lookup()` 的两个分支与 `hit_end` 对齐 |
| TP mismatch | §5 | `effective_tp_size`、`num_sub_keys`、strided addrs |
| 排障 | §6 | 换 TP / 换 block_size / hybrid 各自命中归零的原因 |

---

### 0.5 前置补课：TP 与权重（不熟就先读这一节）

> 这一节是给"没接触过并行推理"的人补的。§2、§5 读不下去时回到这里。

**(1) 权重是什么**

模型 = 结构（代码）+ 权重（几亿~几千亿个数字，训练学出来的）。每层 transformer 的权重（hidden=4096、32 个 Q head、8 个 KV head）：

| 权重 | 形状 | 作用 |
|---|---|---|
| `W_q` | 4096 × 4096 | 输入 → Q |
| `W_k` | 4096 × 1024 | 输入 → K |
| `W_v` | 4096 × 1024 | 输入 → V |
| `W_o` | 4096 × 4096 | 各 head 拼回 hidden |
| MLP 三个 | 4096×14336 等 | 前馈层 |
| LayerNorm γ/β | 4096 | 归一化 |

**K/V 不是权重，是算出来的**：

```
token ─► embedding ─► x（4096 维向量）
                        ├─ x·W_k ─► K（每 head 128 维，每个 token 不同）
                        └─ x·W_v ─► V（每 head 128 维，每个 token 不同）
```

`W_k/W_v` 对所有 token、所有请求是**同一份且只读**；`K/V` 随 token 变。**权重是模具，K/V 是压出来的件；池化搬的是件，不是模具。**

**(2) TP 是什么**

TP = Tensor Parallelism：把同一层的权重切到 T 张卡上同时算，`rank` 是卡号（0..T−1）。

```
TP=1：[   完整 W   ]        TP=4：[W/4][W/4][W/4][W/4]
                                 rank0 rank1 rank2 rank3
```

三条铁律（§2、§5 全靠它）：

| | 切不切 | 为什么 |
|---|---|---|
| 输入 `x` | **不切**，每卡都有完整一份 | 否则算不出自己 head 的 Q/K/V |
| 权重 | **切**（按 head 维） | 每卡只放自己那几个 head 的列 |
| 输出 | **分片** | 每卡只有自己 head 的结果 |

一次前向的真实结构：

```
x（四张卡各一份，内容相同）
 ├─ W_q/W_k/W_v 切片 → 各自 head 的 Q/K/V    ← 无通信
 ├─ attention（head 之间完全独立）            ← 无通信
 ├─ W_o 切片 → partial hidden                ← 无通信
 └─ all-reduce 求和 ─────────────────────────► 完整 hidden   ← 唯一的同步点
                                                │
                                                ▼ 第 1 层（层间串行）
```

- **层内并行，层间串行**：第 k 层的输出是第 k+1 层的输入。
- 每层 2 次 all-reduce（attention 后、MLP 后）。

**(3) 为什么这决定了 KV 必须按 head 分片**

`W_k/W_v` 按 head 列切 → 每卡只有自己那几个 head 的列 → **每卡只能算出自己 head 的 K/V**。所以"同一块的 KV"在物理上是 T 份互不重叠的字节，池化里也就必须是 T 个对象。

**(4) 顺带：TP 主要省的是 prefill 的时间**

| | Prefill（TTFT） | Decode（TPOT） |
|---|---|---|
| 瓶颈 | 算力 FLOPs | 显存带宽（读权重 + KV） |
| TP 切的量 | FLOPs → 1/T | 读取字节 → 1/T |
| 收益 | 接近线性（大矩阵掩盖通信） | 递减（all-reduce 延迟占比上升） |

---

## 1. Key 是怎么拼出来的

### 1.1 半边一：指纹，而且是"链式"的

h_k 编码的不是"第 k 块里有什么"，而是**"第 0 块到第 k 块的整条前缀"**。由此得到两个免费的推论：

**推论 A：粒度可以"合并"，不需要重新 hash。**
`get_block_hashes`（[metadata.py L748-L756](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L748-L756)）在 `group_block_size > hash_block_size` 时不重算，而是套一层惰性视图：

```python
return _LazyGroupedBlockHashList(block_hashes, group_block_size // hash_block_size)
```

`_LazyGroupedBlockHashList.__getitem__` 的实现只有一句（[L836-L839](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L836-L839)）：

```python
# Chained hashes make the final fine-grained hash identify the complete larger block.
return self._block_hashes[(index + 1) * self._scale_factor - 1]
```

**取这一组的最后一个细粒度 hash 当大块的身份。**因为链式结构下它已经蕴含了前面所有内容——不需要把 4 个 hash 再拼一次。

**推论 B：`hash_block_size` 与 `group_block_size` 是两个独立粒度。**前者决定"多久算一次指纹"，后者决定"一组 KV 按多大一块存"。把 `block_size` 从 128 改成 64，fingerprint 链会完全变——这就是 L0 §5.6 第 3 题的答案：**老池子不能命中**，因为 h_k 全变了。

### 1.2 半边二：坐标，七个维度

`KeyMetadata`（[metadata.py L118-L133](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L118-L133)）：

| 字段 | 来源 | 补的是哪个"缺失维度" | 缺了会怎样 |
|---|---|---|---|
| `model_name` | 模型配置 | 权重/分词不同 → KV 不同 | 换模型仍命中 → 输出错 |
| `head_or_tp_rank` | tp_rank（TP mismatch 时是 effective rank） | TP 分片：每个 rank 只存自己那几个 head | 读到别人的 head，**形状对、内容错** |
| `dcp_rank` | DCP 配置 | 上下文并行分片 | DCP 下复用了别人的上下文分片 |
| `pp_rank` | PP 配置 | 流水段：不同 stage 存不同层 | PP 下 A 段拿到 B 段的层 |
| `kv_cache_group_id` | 多 cache group | full / SWA / hybrid 各组 spec 不同 | 不同 spec 的块互相顶替 |
| `cache_role` | `"kv"` / `"state"` | KV 与 Mamba/SWA state 要分开 | state 被当成 KV 读 |
| `cache_family` | 压缩率族 `c1`/`c2`/`mixed` | 同一段 token 的压缩布局不同 | 不同压缩率的块混用 |

> 一句话记法：**指纹管"是不是同一段 token"，坐标管"是不是同一份字节"。**

### 1.3 拼成字符串：三套格式并存

**① 完整 key（默认路径）** —— [L155-L165](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L155-L165)：

```
{model}@dcp:{dcp}@head_or_tp_rank:{tp}@pp_rank:{pp}@group:{g}@cache_role:{role}@cache_family:{family}@{chunk_hash}
```

同一份逻辑在 `_get_key_prefix`（[L366-L388](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L366-L388)）里被**缓存成前缀**，因为同一请求几十个块只有最后一段不同——避免每块都拼一遍。

**② `LayerPoolKey`（老格式，按层一个对象）** —— [L201-L211](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L201-L211)：

```
{model}@dcp:{dcp}@head_or_tp_rank:{tp}@group:{g}@cache_role:{role}@cache_family:{family}@layer_id:{id}@{chunk_hash}
```

**注意：它没有 `pp_rank`。** 推断原因：PP 各 stage 的层区间连续且不重叠，`layer_id` 已经隐含了 stage 归属，再写一次是冗余。这是"老格式靠层号代替段号"的证据，也是 L2 §6.2 那张表里"历史路径多一个 `@layer_id`"的来源。

**③ 后端协议模块（L2 讲的 `block_key` / `gva`）** —— 完全不用 `PoolKey`，由 `bind_layerwise_keys` 自己拼，格式见 `mooncake_layerwise.make_block_key` / `memcache_backend.make_full_key`。

> **读代码时的定位规则**：看到 `@dcp:` 就是完整路径；看到 `@layer_id:` 是老 layerwise 路径；看到 `@pp_rank:` 或 `@pp{N}` 就是后端协议的 block_key/gva 路径。

---

### 1.4 group_id 从哪来：不是推断，是循环下标

启动时按 group 预建一张"坐标模板表"（[pool_worker.py L427-L439](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L427-L439)）：

```python
self.metadata: list[KeyMetadata] = []
for group_id in range(self.num_kv_cache_groups):
    # the mamba kv_heads is not same with the full attention, can't share the cache data
    group_tp_rank = self.tp_rank if self.group_uses_align_state[group_id] else self.head_or_tp_rank
    self.metadata.append(KeyMetadata(..., group_id))     # ← 下标就是 group_id
```

运行时 group_id 是**一路传下来的循环变量**，不是从数据里查出来的：

```
lookup(): for group_id in kv_cache_group_ids:        ← 唯一来源
            → _build_lookup_keys(..., group_id, ...)
            → process_tokens(..., kv_cache_group_id=group_id)
            → _make_key_by_hash(chunk_hash, kv_cache_group_id=group_id)
            → metadata[kv_cache_group_id]（取这一组的模板）
            → PoolKey(KeyMetadata(..., group_id), chunk_hash)
```

见 [metadata.py L420-L443](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L420-L443)。

**为什么要按组存一份模板、而不是一份全局的**：就是上面第 3 行——state/mamba 组不按 head 切（直接用 `tp_rank`），所以**同一个 worker 在不同组里的 `head_or_tp_rank` 取值可以不同**。一份全局坐标表达不了这件事。

---

## 2. 坐标为什么会"多出来"

### 2.1 一个反例：同一块被切成四份

假设 TP=4、`num_kv_heads=8`，那么：

```
逻辑上"第 3 块的 KV" = 8 个 head 的完整张量
物理上：
   rank0 → heads[0:2]   key 里 head_or_tp_rank = 0
   rank1 → heads[2:4]   key 里 head_or_tp_rank = 1
   rank2 → heads[4:6]   key 里 head_or_tp_rank = 2
   rank3 → heads[6:8]   key 里 head_or_tp_rank = 3
```

**"同一块的 KV"物理上根本不是一份，而是 4 份互不重叠的字节。**所以对象数 = `TP × 块数 × 层数`，而 key 必须把 `head_or_tp_rank` 写进去，否则 4 个 rank 会拿同一个 key 写不同的内容。

这是 L0 §6.1 的完整版，也是本讲"坐标即身份"的根据。

### 2.2 关键事实：指纹与并行无关（所以四份共用同一个 hash）

- `block_hashes` 的输入只有 token ids（+ 父 hash + salt），**没有 head、没有 rank**；
- 它在 **scheduler 里对每个请求只算一次**，然后原样下发给每个 TP rank / PP stage。

```python
block_hash_k = hash(block_hash_{k-1}, token_ids[k*B:(k+1)*B], extra_keys)
#                     ↑ 前缀            ↑ 这段 token         ↑ salt / lora
```

```
request(token_ids) ─► scheduler ─► h0..hN        ← 只算这一次
                                    │
      ┌──────────┬──────────┬───────┴───┐
    rank0      rank1      rank2      rank3
   同一份 h   同一份 h    同一份 h   同一份 h
   各自字节   各自字节   各自字节    各自字节
```

**四张卡面对同一个 key，写的是四份不同字节。**

### 2.3 不加坐标会怎样：两处故障，且都不报错

**写侧：覆盖。**

```
rank0 put("…@3f9a") ← heads[0:2]
rank1 put("…@3f9a") ← heads[2:4]   ← 同一个对象名
rank2 / rank3 同理
→ 池里只剩一份（通常是最后写的），或冲突报错
```

**读侧：四张卡都取到同一份。**

```
rank0..3 都 get("…@3f9a") → 拿到同一份字节
→ 至多一张卡是对的，其余三张拿到别人的 head
→ 形状完全正确（都是 128×2×128）→ 不报错，只算错
```

> 措辞要准：不是"读到的不是 rank0 那份"，而是"**所有 rank 读到的是同一份**"。

加坐标之后：

```
rank0 put("…@head_or_tp_rank:0@3f9a")
rank3 get("…@head_or_tp_rank:3@3f9a")   ← 精确取自己那份
```

### 2.4 判据：切过就进 key，没切就不进

| 维度 | 谁切的 | 结果 |
|---|---|---|
| `head_or_tp_rank` | TP 按 head 切 | 每 rank 一份 |
| `pp_rank` | PP 按层切 | 每 stage 一份 |
| `dcp_rank` | DCP 按上下文切 | 每 dcp rank 一份 |
| `group` | 多 cache spec | 每组一份 |
| `cache_role` | KV / state | 两类各一份 |
| `cache_family` | 压缩率 | 每种布局一份 |
| `layer_id` | layerwise 按层拆对象 | 每层一份 |

反例（不切就不进）：**MLA 只有 1 个 latent head，KV 在 rank 间是复制的**，所以 `get_group_tp_size` 里是 `min(tp_size, num_kv_heads) = 1`，只需要一份对象。**坐标维度跟着"切没切"走，不是写死的。**

---

## 3. 三个 block_size，别混

### 3.0 先定义：cache group 是什么

group = **"KV 缓存结构相同的那些层"的集合**。vLLM 分配 KV 时把形状相同、块大小相同的层放一组，共用一套分配器（[`infer_group_block_sizes` L293-L306](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L293-L306)）：

```python
for group in kv_cache_groups:
    block_sizes.append(group.kv_cache_spec.block_size)   # ← 每组一个块大小
```

什么时候会 >1 组（[`uses_hybrid_kv_cache` L284-L290](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L284-L290)：`len(groups) > 1` 且存在非 FullAttention 的 spec）：

| 场景 | 为什么必须分组 |
|---|---|
| 滑动窗口层（Gemma 类） | 只保留最近 W 个 token，形状/长度不同 |
| Mamba / 线性注意力层 | 存的是 **state**，不是 K/V |
| MLA / DSA indexer 等特殊层 | 各有自己的缓存结构 |

**普通模型（Qwen / Llama）只有 1 组**——"多组"是例外，不是常态。但它带来三个后果：key 要带 `group_id`；各组的 `block_size` 可能不同 → 粒度取 LCM；命中长度要所有组都满足 → 取 min。

### 3.1 三个粒度，各管一件事

| 名字 | 谁决定 | 用途 | 取值例子 |
|---|---|---|---|
| `hash_block_size` | vLLM 的 block 哈希粒度 | 算指纹（链长、指纹个数） | 128 |
| `group_block_size` | 每个 cache group 自己的 spec（[`infer_group_block_sizes` L293-L306](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L293-L306)） | 每组按多大一块存 | group0=128、group1=64 |
| `cache_transfer_granularity` | **LCM**（[L325-L333](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L325-L333)） | 命中点/传输必须对齐的粒度 | `lcm(128, 64, …)` |

```python
def infer_cache_transfer_granularity(group_block_sizes, lcm_block_size, kv_cache_group_ids):
    granularities = [lcm_block_size] + [get_group_block_size(group_block_sizes, g) for g in ids]
    return math.lcm(*granularities)
```

**为什么要 LCM**：命中长度要同时对所有 group 都成立（一个组缺就得重算），而各组块大小可能不同。只有 LCM 的整数倍才能让所有组同时"落在块边界上"。这就是代码里到处是 `cache_transfer_granularity` 而不是 `block_size` 的原因。

### 3.2 为什么命中点必须落在 granularity 上：半块没有名字

池子里的对象叫 `…@3f9a`，对应的是**整块 0..127 的 KV**。你没法说"我要这个对象的前 96 个 token"——**96 不在块边界上，没有那个 key**。

```
请求 1000 token，池里前 6 块命中
 ① 1000 向下取整到 128 → 896（7 块可查）
 ② 池子命中 6 块 → 768
 ③ 768 是 128 的倍数 ✅ 可以直接用

若算出来是 800：
   800 / 128 = 6.25 块 ← 半块
 → 只能用到 768，剩下 32 个 token 必须重算
```

为什么不能将就半块：**引擎"跳过 N 个 token"是一个数**（不按组、不按段各一个）；半块会让后续所有块边界错位，`addrs` 指到别的物理块 → 静默错数据。

### 3.3 为什么要取 min：跳过多少只能是一个数

```
group0（full attention，块=128）：命中 768
group1（state / SWA，块=64）    ：命中 512

取 min = 512
  → group0 明明有 512..767 也得丢
  → 否则按 768 跳，group1 的 512..767 是缺的 → attention 读到未初始化显存
```

**宁可少命中，不能有一组缺**——这就是 `return min(hits)` 的全部理由。

### 3.4 尾块怎么办

`get_partial_block_index`（[L759-L777](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L759-L777)）负责识别"是否存在需要单独搬运的尾块"——注意它返回的是**要搬的那一块的索引**，不是"要把命中长度减到对齐"。

> 分清两件事：**命中判定**按块对齐（半块没有 key）；**传输**在 `discard_partial_chunks=False` 时可以把尾块一起搬（L0 §6.3）。

---

## 4. 命中判定：`lookup()` 到底做了什么

### 4.1 主体（[pool_worker.py L3784-L3850](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3784-L3850)）

```python
for group_id in kv_cache_group_ids:
    keys, starts, ends = self._build_lookup_keys(token_len, block_hashes, group_id, use_layerwise)
    res = self.m_store.exists(keys)
    if use_layerwise:
        res = self.check_all_layers_exists(res, num_layers)      # ① 每层都在才算命中
    if group_uses_align_state[group_id]:                         # ② 对齐状态组（Mamba align）
        hit_end = 0
        for index in range(len(ends) - 1, -1, -1):               #    从后往前找"最远的可对齐命中"
            if res[index] == 1 and ends[index] % self.cache_transfer_granularity == 0:
                hit_end = ends[index]; break
    else:                                                        # ③ 普通组：找第一个断点
        hit_end = ends[-1]
        for index, value in enumerate(res):
            if value != 1:
                hit_end = 0
                for hit_index in range(index, 0, -1):            #    回退到能对齐的边界
                    if starts[hit_index] % self.cache_transfer_granularity == 0:
                        hit_end = starts[hit_index]; break
                break
    hits.append(hit_end)
return min(hits) if hits else 0
```

三个要点：

1. **`use_layerwise` 时"每层都在"才算这块命中**（`check_all_layers_exists`）——因为 layerwise 的对象是"一块 × 一个 head 的整层栈"，缺一层就是残缺对象。
2. **两个分支的对齐方向相反**：
   - 普通组：从**前**往后找第一个断点，再**往回**退到最近的粒度边界（宁可少算，不能多算）；
   - align 状态组：从**后**往前找最远的"命中且对齐"点（因为状态组的块语义不同，断点不一定是最后）。
3. **多 group 取 `min(hits)`**：任何一个 group 缺，就得从那里重算。

### 4.2 key 的展开：一个块 → N 个 key

`_build_lookup_keys`（[L3752-L3782](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3752-L3782)）只生成"本 rank 视角"的 key；真正查池子前还要展开成**全 rank**（[`_expand_lookup_keys_by_rank` L3880-L3889](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3880-L3889)）：

```
for pp_rank in range(pp_size):
  for dcp_rank in range(dcp_size):
    for head_or_tp_rank in range(num_head_or_tp_ranks):   ← 由 get_group_tp_size() 决定
      for key in keys: ...
```

**为什么要展开**：一个块要被判定为"可用"，必须**所有 writer 都写过**。TP=4 时只查自己 rank 的 key，就可能"自己写过、别人没写过"却判定为命中 → 加载时缺 head。所以命中判定查的是"**全体 rank × 全体 PP 段**"的交集。

`num_head_or_tp_ranks` 由 `get_group_tp_size` 决定（[L3859-L3866](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3859-L3866)）：

```python
if self.use_kvpp:            return tp_size * pcp_size
if self.tp_mismatch:         return effective_tp_size      ← 统一到 max(TP)
if group_uses_align_state:   return tp_size
return min(tp_size, _get_group_num_kv_heads(group_id))     ← MLA 时 num_kv_heads=1
```

最后一行解释了一个常见困惑：**MLA / 单 head 场景下 `min(tp, 1) = 1`**，因为 KV 在 TP rank 之间本来就是复制的，只需要一份对象。

### 4.3 scheduler 侧的同款判定

`_get_store_lookup_hit_tokens`（[pool_scheduler.py L318-L359](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L318-L359)）是同一套逻辑的 scheduler 版本，差别只有两点：

- `use_layerwise` 时**从 block 0 查起**（L326-L328），因为远端池子里的分层对象可能和本地 prefix cache 的起点不一致；
- 它的三值判定（全 1 / 有 0 / 其他报错）就是 L2 §4.3 讲过的那段——**这正是"读路径 fail-closed"的落点**。

> 注意 `lookup()` 里也有一个 `try/except → return 0`（[L3842-L3849](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3842-L3849)）：worker 侧的兜底是"当成没命中"，而 scheduler 侧的 `exists` 判定是"报错"。两处不一致是**分层职责**：worker 只负责"我这儿能给出多少"，最终决策仍在 scheduler。

---

## 5. TP mismatch：把坐标空间统一到 `max(TP)`

### 5.1 判定与算出来的量（[metadata.py L80-L113](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L80-L113)）

```python
effective_tp_size = max(local_tp_size, peer_tp_size)
enabled = (
    peer_tp_size != local_tp_size
    and not use_mla          # MLA 只有 1 个 latent head，无 head 可分
    and not use_hybrid
    and num_kv_heads >= effective_tp_size
    and num_kv_heads % effective_tp_size == 0     # 必须整除
)
local_heads_per_rank     = num_kv_heads // local_tp_size
effective_heads_per_rank = num_kv_heads // effective_tp_size
num_sub_keys             = local_heads_per_rank // effective_heads_per_rank
```

例：P(TP=8)、D(TP=4)、`num_kv_heads=8` →

| 量 | 值 |
|---|---|
| `effective_tp_size` | 8 |
| `local_heads_per_rank`（D 侧） | 2 |
| `effective_heads_per_rank` | 1 |
| `num_sub_keys` | 2 |

**任一条件不满足 → `enabled=False`，且不报错**，退化成"各算各的 key 空间"，表现就是命中率归零。所以"TP 不一致 + MLA"这个组合，池化必然 0 命中。

### 5.2 怎么写 key（[pool_worker.py L3469-L3507](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3469-L3507)）

```python
for sub_idx in range(self.num_sub_keys):
    effective_rank = self.tp_rank * self.num_sub_keys + sub_idx     # D 侧 rank0 → 0,1
    addrs, sizes = self._build_strided_addrs(block_id, token_count, sub_idx)
    all_keys.append(self._make_sub_key_str(base_key, effective_rank))
```

于是 P(TP8) 和 D(TP4) 看到的都是"**每个 effective rank 一个对象**"，布局一致 → 可互通。

**为什么这叫"统一坐标"而不是"格式转换"**：它没有做任何数据形态转换（并非把"2 head 的对象"改写成"1 head 的对象"），而是**把所有 rank 重新编号到同一个坐标系**，让两侧对"一个对象是什么"达成一致：

| | 自然粒度 | 统一后的写法 |
|---|---|---|
| P（TP=8） | 1 head / 卡 | key rank ∈ 0..7（无需拆分） |
| D（TP=4） | 2 heads / 卡 | rank0→{0,1}、rank1→{2,3}、rank2→{4,5}、rank3→{6,7} |

两边都变成"**每 effective rank 一个对象**"，读写才可比。

**不统一会怎样**：D 侧 rank1 拿 key `rank:1` 去读，读到的是 P 侧 rank1 存的 **1 个 head**，而它需要 **2 个 head** → 形状对不上或只填一半，**且不报错**。

**`_build_strided_addrs` 的代价**（[L3462-L3466](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3462-L3466)）：

```python
for t in range(token_count):
    addrs.append(block_base + t * entry_per_token_bytes + head_offset_bytes)
    sizes.append(self.sub_size_bytes)
```

**每个 token 一个描述符**——因为 D 侧的那个 sub-key 只覆盖 1 个 head，而它在显存里是按 token 跨步分布的。量级：44 层 × 2 张量 × 128 token ≈ **11k 个描述符 / 每块每 sub-key**。这就是 TP mismatch 属于"能用但很贵"的路径、且 L2 里要强调 `addrs` 是二维的原因。

### 5.3 两个工程细节

- **旋转**：`offset = self.tp_rank % len(keys)` 后把 key/addrs/sizes 一起旋转（[L3521-L3525](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3521-L3525)）。目的：让不同 rank 从不同对象开始取，避免整机同时打同一个热点对象。
- **字符串手术刀**：`_replace_key_field`（[L3868-L3878](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L3868-L3878)）可以在**不重建 key** 的情况下替换某个 `@field:` 的值（典型就是换 `head_or_tp_rank`）。改坐标不改指纹的最小操作。

---

## 6. 三个排障场景

| 现象 | 根因 | 先看哪里 |
|---|---|---|
| **换 TP 后命中率归零** | `infer_tp_mismatch_info` 的 `enabled` 为假（MLA / hybrid / 不整除 / heads 不足）→ key 空间静默退化 | 打 `enabled / effective_tp_size / num_sub_keys`，再比对两侧 `num_head_or_tp_ranks` |
| **改 `block_size` 后老池子全 miss** | 指纹链长度变了，h_k 全变 | `hash_block_size` 是否随配置一起改；`get_block_hashes` 的 scale |
| **hybrid 模型命中率天生偏低** | 多 group，`min(hits)` 取最小；`granularity` 是各组 LCM | 各组 `group_block_size` 与 `cacheable_group_ids` |

---

## 7. 自检 5 题

1. `block_size=128`、请求 1000 token，池子里 h0~h5 有、h6 没有。`lookup()` 返回多少？如果 h3 缺、h4~h6 都有呢？
2. `LayerPoolKey.to_string()` 里为什么没有 `pp_rank`？什么情况下这个省略会出问题？
3. TP=8 的 P 节点、TP=4 的 D 节点，`num_kv_heads=8`：`num_sub_keys` 是多少？D 侧 rank1 会写哪几个 effective rank？为什么每块描述符会涨到 1 万量级？
4. 一个模型有 group0(block=128) 与 group1(block=64)，`granularity` 是多少？为什么不能用 64？
5. 命中判定为什么要展开到"全体 rank × 全体 PP 段"，而不是只查自己 rank 的 key？

---

## 8. 待验证点

1. 全部结论来自源码阅读，**未在 NPU + Mooncake/Memcache 真机验证**。
2. §1.3 中"`LayerPoolKey` 省略 `pp_rank` 是因为 `layer_id` 已隐含 stage"是**推断**，未找到设计文档原文。
3. §5.2 的描述符量级（≈11k/块/sub-key）是按 44 层、128 token 粗算，实际随模型形状与 `discard_partial_chunks` 变化。
4. worker 侧 `lookup()` 的 `try/except → 0` 与 scheduler 侧"报错"的不一致，是否为有意分层（§4.3），尚待与作者确认。
