# LLM 推理性能与显存 · 学习文档

> **文档来源**：本文件由三部分合并而成
> 1. 课堂/对话知识梳理（注意力机制、显存构成、激活值、Mask、KV Cache 容量与并发）
> 2. 原笔记 `LLM_推理性能学习笔记.md`（延迟指标、吞吐、goodput、batch、chunked prefill、调度器参数）
> 3. 延伸补充（标注 🔍），含推导、数字、工程细节与常见误区
>
> **配套文件**：`TODO.md`（待办清单）、`notes/推理培训-课堂记录.md`（课堂过程记录）
>
> **阅读建议**：想快速上手看第 0 章；想理解"为什么慢"看第 2、4 章；想理解"为什么 OOM"看第 3 章；想调参看第 5 章。

---

## 0. 一页纸速查

| 概念 | 一句话 | 关键公式 |
|---|---|---|
| **Prefill** | 一次性处理整个 prompt，算力密集 | FLOPs ≈ 2·P·S |
| **Decode** | 每步生成 1 个 token，访存密集 | 每步读 P + B·KV |
| **KV Cache** | 历史 token 的 K/V，避免重算 | 每 token = 2·L·h_kv·d_h·bytes |
| **TTFT** | 首 token 前的等待 | 排队 + prefill + 首次 decode |
| **TPOT** | 平均每 token 间隔 | (总时延 − TTFT)/(输出 token − 1) |
| **ITL** | 相邻 token 实际间隔 | 受混批、抢占影响 |
| **Throughput** | 总产出 | 总 token / 总时间 |
| **Goodput** | SLO 内的有效产出 | 吞吐 × 达标率 |
| **显存** | 权重 + KV + 激活 + 框架开销 | 见 3.1 |
| **并发上限** | 由 KV 池决定 | 可用显存 / (KV/token × 平均长度) |

**一句话总纲**：推理性能 = 在「显存容量」约束下，用「算力」和「带宽」两种资源，在「延迟」与「吞吐」之间做权衡。

---

## 1. 模型结构：注意力与 KV Cache 的来源

### 1.1 一次前向的数据流

以 Llama 风格 decoder-only 模型为例，形状记作 `[B, S, H]`：

```
token ids            [B, S]
  ↓ embedding
hidden states h      [B, S, H]        ← 残差流，逐层传递
  ↓ RMSNorm → Q/K/V 投影
Q, K, V              [B, S, H]        ← K/V 会被写入 KV Cache
  ↓ 注意力打分
attn scores          [B, h, S, S]     ← 不优化则物化，O(S²)
  ↓ softmax（加 mask）→ 加权求和
attn out             [B, S, H]
  ↓ O 投影 + 残差
h'                   [B, S, H]
  ↓ RMSNorm → MLP 上投影（SwiGLU 双路）
gate, up             [B, S, I]        ← I ≈ 3~4H，激活峰值大户
  ↓ 逐元素乘 → 下投影 + 残差
下一层 h             [B, S, H]
  ...
final hidden         [B, S, H]
  ↓ lm_head
logits               [B, S, vocab]    ← 又一个 O(S·V) 的大户
```

**关键认识**：`hidden states` 是沿残差流传递的主向量；其余（scores、gate/up、logits）都是临时激活值，算完即弃。而 **K/V 因为后续 decode 要反复使用，被"扣留"下来，就成了 KV Cache**。

> 🔍 **延伸**：KV Cache 本质上是"被持久化的激活值"。理解这一点，就能明白为什么 GQA / MLA 这些"压缩 KV"的技术，实际上是在压缩一份激活值。

### 1.2 MHA（Multi-Head Attention）

最经典的注意力，2017 年 Transformer 原论文提出。

- 输入分别乘 $W_Q, W_K, W_V$ 得到 Q/K/V，按 head 数切分
- 每个 head 独立计算：$\text{softmax}(Q_iK_i^T/\sqrt{d_k})V_i$
- 所有 head 结果拼接后再做一次输出投影

**代价**：每个 token 都要缓存全部 head 的 K/V。

$$\text{KV}_{token} = 2 \times L \times h \times d_h \times \text{bytes}$$

### 1.3 GQA / MQA（工程上最常用的折中）

| 变体 | KV head 数 | 说明 |
|---|---|---|
| **MHA** | $h_{kv} = h$ | 每个 Q head 独享一组 KV |
| **GQA** | $h_{kv} = h/g$ | $g$ 个 Q head 共享一组 KV，主流选择（Llama-3、Qwen2.5） |
| **MQA** | $h_{kv} = 1$ | 所有 head 共享一组 KV，压缩最狠但质量损失明显 |

实现上，GQA 在注意力计算前用 `repeat_kv` 把 KV 沿 head 维复制 $g$ 次，让形状对齐，**计算图与 MHA 完全一致**，只是缓存变小。

> 🔍 **延伸**：`repeat_kv` 是"逻辑复制"——显存里仍只有一份，是 `expand` 出来的视图。真正的收益在 KV Cache 的存储量。

### 1.4 MLA（Multi-head Latent Attention，DeepSeek-V2/V3）

**核心思想**：不缓存每个 head 的完整 K/V，而是缓存一个**低维潜在向量**，推理时再解压。

**（1）低秩压缩**

$$c_t^{KV} = W^{DKV} h_t \quad (d_c \ll h \cdot d_h)$$
$$k_t^C = W^{UK} c_t^{KV}, \qquad v_t^C = W^{UV} c_t^{KV}$$

只缓存 $c_t^{KV}$（维度 $d_c$），K/V 在需要时从它投影出来。

**（2）解耦 RoPE（decoupled RoPE）**

RoPE 是位置相关的旋转，**不能与低秩投影交换顺序**，所以压缩后的 K 无法直接套 RoPE。MLA 的解法是额外维护一份小的 RoPE key：

$$k_t^R = \text{RoPE}(W^{KR} h_t) \quad (d_R \ll d_h)$$

最终的 key 是拼接：$k_t = [k_t^C; k_t^R]$（每个 head 各自拼接）。

**（3）缓存量**

$$\text{KV}_{token}^{MLA} = L \times (d_c + d_R) \times \text{bytes}$$

DeepSeek-V2：$d_c = 512$、$d_R = 64$，即每层每 token 只存 **576** 个元素。相比同规模 MHA 小一到两个数量级；论文报告相比前代降低约 **93%**。

**（4）矩阵吸收（推理加速的关键）**

注意力打分里：

$$q_t^T k_s = (W^Q h_t)^T (W^{UK} c_s) = \underbrace{(W^{UK\,T} W^Q h_t)}_{\tilde q_t}^T c_s$$

也就是说，**可以把 $W^{UK}$ 提前吸收进 $W^Q$**，直接用 $\tilde q$ 与潜在向量 $c_s$ 做点积 —— 推理时根本不需要把 K 显式解压出来。同理 $W^{UV}$ 可以吸收进 $W^O$。

> 🔍 **这是 MLA 真正的杀手锏**：不只是缓存小，还省掉了"解压"这一步计算。但注意——**这只对不带 RoPE 的内容部分成立**，RoPE 部分必须单独算，这正是解耦 RoPE 存在的第二个理由。

### 1.5 四者对比

| | MHA | MQA | GQA | MLA |
|---|---|---|---|---|
| KV head 数 | $h$ | 1 | $h/g$ | 潜在向量 + 小 RoPE key |
| 每 token KV | $2Lhd_h$ | $2Ld_h$ | $2L(h/g)d_h$ | $L(d_c+d_R)$ |
| 质量 | 最好 | 有损 | 接近 MHA | 接近/超过 MHA |
| 推理成本 | 高 | 最低 | 低 | 低 |
| 代表模型 | Llama-2 | PaLM | Llama-3、Qwen2.5 | DeepSeek-V2/V3 |

**本质区别**：GQA/MQA 是"**共享** KV head"；MLA 是"**压缩再解压**"。两者正交，可以叠加。

### 1.6 每 token KV 实测对比（FP16）

| 模型 | 层数 | KV 头 | head_dim | 每 token KV |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 28 | 4 | 128 | **56 KB** |
| Llama-3-8B | 32 | 8 | 128 | **128 KB** |
| Llama-3-70B | 80 | 8 | 128 | **320 KB** |
| Llama-2-70B（MHA） | 80 | 64 | 128 | **2.5 MB** |
| DeepSeek-V2（MLA） | 60 | — | $d_c$=512, $d_R$=64 | **≈ 67.5 KB** |

### 1.7 Mask：控制信息可见性

**定义**：注意力里的"可见性表"，作用在 softmax 之前，把不允许的位置压成 $-\infty$。

$$\text{Attn} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V, \quad M_{ij} \in \{0, -\infty\}$$

| 类型 | 规则 | 用途 |
|---|---|---|
| **Causal**（因果/下三角） | $i$ 可见 $j \le i$ | 防偷看未来，训练可并行 |
| **Padding** | 屏蔽 pad 位 | batch 变长序列 |
| **Document**（块对角） | 同文档内可见 | 长文本 packing |
| **Sliding window** | 只看 $[i-w, i]$ | 降 $O(S^2)$ |
| **Prefix LM** | 前缀双向 + 后缀因果 | T5 / GLM |
| **Tree** | 按树结构可见 | 投机解码 |

**工程要点**：

```python
# 加法掩码（推荐，便于合并多种 mask）
mask = torch.triu(torch.full((S, S), torch.finfo(dtype).min), diagonal=1)
scores = scores + mask          # 必须在 softmax 之前
```

- 用 `finfo.min` 而非 `-inf`：避免整行被 mask 时 softmax 分母为 0 → NaN
- 现代 kernel 用 `is_causal=True` **不物化** $S\times S$ 矩阵（否则 S=32768 时 bool 掩码就要 1 GB）
- **chunked prefill / 投机解码** 的因果掩码必须按**绝对位置偏移**构造，是经典的 bug 来源

---

## 2. 推理过程：一次请求经历什么

### 2.1 请求生命周期

```text
请求到达 ── 排队 ── Prefill(prompt) ── 首次 decode ── decode ── decode ── ...
                        │                    │              │
                        └── TTFT 的主要部分 ─┘              └── TPOT / ITL
```

1. **排队（queueing）**：等调度器与设备资源
2. **Prefill**：整个 prompt 过一遍模型，算出各层 K/V 写入 KV Cache
3. **首次 decode**：基于 prompt 的 KV 生成第 1 个输出 token
4. **后续 decode**：每步生成 1 个 token 并追加到 KV Cache，直到 EOS 或 max_tokens

自回归模型不会一次生成全部输出，而是重复很多个 decode step。

### 2.2 Prefill 与 Decode 的本质差异

| 维度 | Prefill | Decode |
|---|---|---|
| 每次处理的 token 数 | $S$（整个 prompt） | 1（每请求） |
| 矩阵形状 | 大 GEMM，$S \times d$ | 小 GEMM，$1 \times d$ |
| 主要瓶颈 | **算力** | **显存带宽** |
| 激活值 | 大（$B\cdot S$） | 极小 |
| KV Cache | 写入 | 读取并追加 |
| 典型耗时 | 几十 ms ~ 几秒 | 几 ms ~ 几十 ms |

> 🔍 **这个差异是后面所有优化的根源**：chunked prefill 是在管 prefill 的"独占性"，PD 分离是在管两种负载的资源冲突，KV 量化是在管 decode 的带宽压力。

### 2.3 计算特征：Roofline 视角

定义**算术强度** = FLOPs / Bytes。硬件的"脊点"（ridge point）：

$$\text{ridge} = \frac{\text{峰值 FLOPS}}{\text{HBM 带宽}}$$

| 硬件 | FP16 FLOPS | HBM 带宽 | ridge |
|---|---:|---:|---:|
| A100 80G | 312 TFLOPS | 2.0 TB/s | ≈ 156 |
| H100 SXM | 990 TFLOPS | 3.35 TB/s | ≈ 295 |

**Prefill**：每字节权重参与 $S$ 个 token 的计算，算术强度 ≈ $S$，通常 $S \gg$ ridge → **算力受限**。

**Decode**：每步读一遍权重只算 batch 个 token，算术强度 ≈ $2B$ → **小 batch 时带宽受限**。

> 🔍 **量化会移动脊点**：权重字节数减半，decode 的算术强度翻倍，脊点对应的 batch 减半。所以 INT4 推理在更小的 batch 下就从带宽受限转为算力受限。

**吞吐上限的三种情形**（重要修正）：

| 场景 | 主导资源 | 吞吐近似 |
|---|---|---|
| 小 batch decode | HBM 带宽（读权重） | $\approx B \cdot \text{BW} / P_{\text{bytes}}$（随 batch 线性增） |
| 大 batch decode | 算力 | $\approx F / (2P)$（饱和，与 batch 无关） |
| 超长上下文 | HBM 带宽（读 KV） | $\approx \text{BW} / \text{KV}_{token}$ |

> 🔍 **注意**：常听到的"吞吐上限 ≈ 带宽 / 每 token KV 大小"**只在 KV 读取量超过权重读取量时成立**（超长上下文、超大 batch）。中短上下文下，真正的饱和点是**算力**，即 $F/(2P)$。

**算例**（8B 模型，A100 80G，FP16）：

- 权重读取：$16\,\text{GB} / 2\,\text{TB/s} = 8\,\text{ms}$（与 batch 无关）
- 每 token 算力：$2P/F = 16\times10^9 / 312\times10^{12} = 51\,\mu s$
- 打平点：$B^* = b \cdot F/(2\text{BW})$，FP16 时 $b=2$ → **≈ 156**（batch 超过它就进入算力受限区）
- 理论上限：$F/(2P) \approx 1.95\times10^4$ tok/s，实际通常只有 **20%~40%**

---

## 3. 显存：推理系统最硬的约束

### 3.1 显存构成清单

| 类别 | 内容 | 性质 |
|---|---|---|
| **模型权重** | 参数矩阵、embedding、lm_head、norm、MoE 专家、量化 scale | 常驻，最大项 |
| **KV Cache** | block 池、prefix cache、block table | 常驻（池化） |
| **CUDA Context** | 驱动、kernel 二进制、cuBLAS/cuDNN 句柄 | 常驻，300~600 MB，**不可调** |
| **CUDA Graph 池** | 捕获 decode 计算图用的 private pool，按 batch 档位各一份 | 常驻，几百 MB |
| **通信 buffer** | NCCL all-reduce / all-to-all 缓冲 | 常驻（多卡），几百 MB |
| **激活值** | hidden states、MLP 中间层、attention scores、logits | **瞬时**，峰值决定 OOM |
| **算子 workspace** | FlashAttention scratch、cuBLAS workspace | 瞬时 |
| **分配器开销** | 已预留未使用、外部碎片 | 不可见，10%~20% |

### 3.2 激活值：被低估的峰值杀手

**定义**：前向计算中产生、算完即弃的临时张量。主向量 = hidden states。

**峰值出现在三处**：

| 位置 | 张量 | 大小 |
|---|---|---|
| MLP 上投影 | $[B,S,I]\times 2$ | $2BSI \cdot$ bytes |
| Attention scores（未优化） | $[B,h,S,S]$ | $BhS^2 \cdot$ bytes |
| Logits | $[B,S,V]$ | $BSV \cdot$ bytes |

**算例**（Llama-3-70B，prefill 4096 token，BF16）：

| 张量 | 计算 | 大小 |
|---|---|---:|
| hidden states | $4096\times8192\times2$ | 67 MB |
| MLP gate + up | $2\times4096\times28672\times2$ | **470 MB** |
| attention scores | $64\times4096\times4096\times2$ | **2.15 GB** ⚠️ |
| logits | $4096\times128256\times2$ | **1.05 GB** ⚠️ |

**这解释了三个设计**：

- **FlashAttention**：不物化 $[B,h,S,S]$，改分块计算 + online softmax（只保留 running max/sum 和输出累加器）→ 2.15 GB 直接消失
- **Chunked prefill**：切小 $S$，压住 MLP 和 logits 峰值
- **`gpu_memory_utilization` 不能设 1.0**：要给激活峰值留余量（vLLM 启动时跑假请求实测）

> 🔍 **Online softmax 为什么可行**：softmax 需要全局 max，但可以增量维护 $m_i$（已见最大）和 $\ell_i$（已见指数和），每来一个新块就用 $e^{m_{old}-m_{new}}$ 修正旧累加器。数学上与一次性计算完全等价，因此 FlashAttention 是**精确**的，不是近似。

### 3.3 KV Cache 容量计算

**每 token**：

$$\text{KV}_{token} = 2 \times L \times h_{kv} \times d_h \times \text{bytes} \quad (\text{MLA 用 } L(d_c+d_R))$$

**并发上限**：

$$\text{max\_concurrency} \approx \frac{\text{可用显存}}{\text{KV}_{token} \times \text{平均上下文长度}}$$

**算例 1**：Llama-3-8B，单卡 80 GB，留 60 GB 给 KV

| 上下文 | 每请求 KV | 可并发 |
|---|---:|---:|
| 4K | 512 MB | ≈ 117 |
| 16K | 2 GB | ≈ 30 |
| 32K | 4 GB | ≈ 15 |
| 128K | 16 GB | ≈ 3 |

**算例 2**：Llama-3-70B，8×H100（TP=8，约 480 GB 给 KV）

| 上下文 | 每请求 KV | 可并发 |
|---|---:|---:|
| 8K | 2.6 GB | ≈ 183 |
| 32K | 10.5 GB | ≈ 45 |

**结论**：长上下文贵，不是算力不够，而是**显存被 KV 直接吃光，并发掉到个位数**。

### 3.4 显存管理：为什么不会"越堆越多"

**核心事实**：生产引擎（vLLM）在**启动时一次性预分配 KV 池**，之后只在池内分配/回收，**不会随请求增长向系统要显存**。所以你看到的显存曲线是平的，满了触发的是调度行为而非 OOM。

**四道闸门**：

| 闸门 | 机制 | 参数 |
|---|---|---|
| ① 请求结束回收 | block 引用计数归零即归还 | EOS / max_tokens |
| ② 准入控制 | 分配不到 block 就进 waiting 队列 | `block_manager.can_allocate()` |
| ③ 抢占 preemption | 丢弃/换出在跑请求的 KV | V0: swap 到 CPU 或 recompute；V1: 以 recompute 为主 |
| ④ 并发硬上限 | 同时在跑的序列数封顶 | `max_num_seqs` |

**PagedAttention 的三件套**：

- `BlockAllocator`：管理物理 block 池（默认 `block_size=16` token）
- `BlockSpaceManager` / `BlockTable`：维护"逻辑块 → 物理块"的映射
- 引用计数 + **copy-on-write**：beam search / `n>1` 采样时共享前缀，只在分叉时复制

**为什么 block 大小是 16**：太小 → block table 变长、寻址开销大、kernel 效率低；太大 → 内部碎片（最后一个 block 浪费）。16 是经验平衡点。

**传统实现的浪费**：按 `max_len` 预留，导致 **60%~80%** 的 KV 显存被浪费（vLLM 论文数据）。PagedAttention 的价值就在这。

### 3.5 分配器与碎片

- `torch.cuda.memory_allocated()` ≪ `memory_reserved()` 是常态：caching allocator 预留但不归还
- 外部碎片：不同大小的张量反复分配释放，"明明显存够却 OOM"
- `expandable_segments:True`：允许虚拟地址连续、物理不连续的段扩展，缓解碎片
- `torch.cuda.empty_cache()`：只是把空闲缓存还给驱动，**不能释放仍在使用的张量**

### 3.6 容量规划与排队论

**Little's Law**：

$$L = \lambda \times W$$

- $L$：稳态并发数（= 占用 KV 份数）
- $\lambda$：到达率（req/s）
- $W$：单请求平均驻留时间（s）

**例**：20 req/s × 5 s = 100 并发。若容量 117，刚好扛住；若来 30 req/s → 150 > 117 → 排队、TTFT 上升、抢占增多，直到客户端超时或网关限流把 $\lambda$ 压回去。

**满载时的现象序列**：

1. TTFT 上升（waiting 队列变长）
2. TPOT 抖动（抢占/重算）
3. 吞吐进入平台期（资源饱和）
4. 429 / 超时（网关限流）

---

## 4. 性能指标

### 4.1 TTFT（Time To First Token）

$$TTFT = \text{排队时间} + \text{Prefill 时间} + \text{首次 decode 时间}$$

衡量"系统有没有开始回应"。长 prompt、系统繁忙、调度不合理都会增大它。

> 健康形态参考：TTFT ≈ 800 ms，TPOT ≈ 40 ms/token。

### 4.2 TPOT / ITL

- **TPOT（Time Per Output Token）**：平均每 token 耗时，$TPOT = \dfrac{\text{总时延} - TTFT}{\text{输出 token 数} - 1}$
- **ITL（Inter-Token Latency）**：相邻 token 的实际到达间隔

$TPOT$ 是统计均值，$ITL$ 是逐 token 的用户体验。通常 $TTFT > TPOT$。

但**高负载下 TPOT/ITL 可能显著变大**，甚至接近或超过 TTFT：decode 被新请求 prefill 干扰、大 batch 资源饱和、抢占重算。

### 4.3 吞吐

$$Throughput = \frac{\text{所有请求生成的 token 总数}}{\text{时间}} \quad (\text{tokens/s})$$

回答"这台服务总体能产多少 token"，**不保证单个用户快**。可以靠攒大 batch 做出很高总吞吐，但排后面的用户可能很久看不到首 token。

> 比喻：大巴坐满再发车，总体运力高，但先到的人要等。

### 4.4 Goodput（有效吞吐）

**SLO 内完成、对用户真正有价值的吞吐**。

例：SLO 要求 TTFT ≤ 1 s、TPOT ≤ 50 ms，总吞吐 10,000 tok/s，达标率 90%：

$$Goodput \approx 10{,}000 \times 90\% = 9{,}000\ \text{useful tokens/s}$$

不同压测工具定义略有差异（达标请求/s 或达标输出 token/s），核心不变：**超出延迟 SLO 的工作不算有效产出**。

> 🔍 **为什么 goodput 才是优化目标**：单纯最大化吞吐的极端解是"无限攒 batch"，此时吞吐很高但 TTFT 爆表，goodput 反而下降。goodput 是同时约束了延迟和吞吐的复合指标，天然对应业务价值。

### 4.5 指标间的关系

| 关系 | 说明 |
|---|---|
| $TTFT \approx$ 排队 + prefill | 与 prompt 长度**线性相关**（$O(S)$ 计算，但 kernel 效率使实际超线性） |
| $TPOT \approx$ 单 step decode 耗时 | 但受同批次混入的 prefill、batch 大小影响 |
| 吞吐 $\propto$ batch（未饱和时） | 饱和后进入平台期 |
| 抢占 $\uparrow$ → TPOT 尾部 $\uparrow$ | 重算表现为一次超长 ITL |

---

## 5. 调度：在有限资源里做取舍

### 5.1 连续批处理（Continuous Batching）

不等整个 batch 全部结束，**任一请求结束就立刻补一个新请求进来**，按 step 粒度调度。这是现代推理引擎吞吐远高于朴素 batching 的根本原因。

> 🔍 **对比**：静态 batching 下，一个 batch 里最长的请求决定整批完成时间，短请求的算力被浪费（padding）。连续批处理消除了这种浪费。

### 5.2 Chunked Prefill

**思想**：把长 prompt 切成多个 chunk，分散到多个调度 step 执行。例如 4000 token 的 prompt，每轮最多 prefill 1000 token。

```text
非 chunked：prefill 4000 ── 首次 decode ── 首 token
chunked：   prefill 1000 ─ 其他请求 ─ prefill 1000 ─ ... ─ 首次 decode ─ 首 token
```

**代价**：自己的 TTFT 可能变长

$$TTFT = \text{排队} + \sum\text{所有 chunk 计算} + \text{chunk 间等待} + \text{首次 decode}$$

**收益**：

- 长 prompt 不再一次性阻塞所有正在 decode 的请求
- 存量请求的 TPOT/ITL 更稳定
- 整体 SLO 达标率与 goodput 往往更好

若设备空闲、或调度器连续执行同一请求的所有 chunk，chunked prefill 不一定增加 TTFT；影响主要在并发负载下体现。

### 5.3 调度器伪代码与参数

```python
def schedule():
    token_budget = max_num_batched_tokens

    # 第一优先：running 请求的 decode
    for req in running:
        n = 1 + len(req.spec_token_ids)          # 投机解码时 >1
        allocate_kv_blocks(req, n)
        token_budget -= n

    # 第二优先：waiting 请求的 prefill，可分 chunk
    while waiting and token_budget > 0 and len(running) < max_num_seqs:
        req = waiting.peek()
        n = min(req.remaining_prompt_tokens, token_budget, chunk_limit)
        if not allocate_kv_blocks(req, n):
            break
        token_budget -= n

    return SchedulerOutput(...)
```

| 参数 | 含义 | 约束什么 |
|---|---|---|
| `max_num_batched_tokens` | 单轮最多处理的 token 总数 | 一轮前向的总工作量 |
| `token_budget` | 本轮剩余 token 名额 | 调度过程中递减 |
| `max_num_seqs` | 同时服务的最大请求数 | 并发规模、KV 占用、调度开销 |
| `chunk_limit` | 单请求单轮最多 prefill 的 token 数 | 长 prompt 对 batch 的占用 |
| `remaining_prompt_tokens` | 该请求剩余未 prefill 的 token 数 | 请求自身剩余工作 |

**数值例子**：`max_num_batched_tokens=4096`，running=100（各占 1 token），`chunk_limit=1024`，新请求 prompt=10000：

```text
token_budget = 4096 - 100 = 3996
n = min(10000, 3996, 1024) = 1024
```

本轮只 prefill 1024 token，剩 8976 留待后续。

> ⚠️ **`max_num_batched_tokens` 与 `max_num_seqs` 是两个维度**：前者限制"本轮处理多少 token"，后者限制"同时服务多少请求"。即使 token budget 有余，只要到达 `max_num_seqs`，调度器也可能不再接纳新请求。

### 5.4 抢占（Preemption）

池满时的最后手段：

| 方式 | 做法 | 代价 |
|---|---|---|
| **Recompute** | 丢弃 KV，之后重新 prefill | 浪费算力；表现为一次超长 ITL |
| **Swap** | 把 KV 换出到 CPU 内存，需要时换回 | 占用 PCIe 带宽，延迟高 |

V0 两条路径都有；V1 以 recompute 为主（swap 需要额外的 CPU 内存管理和传输开销，收益不稳定）。

> 🔍 **为什么 recompute 常常更优**：swap 需要把 KV 写出去再读回来，PCIe 带宽只有 ~32 GB/s，而重新 prefill 走的是 HBM + 算力。短上下文时重算更便宜；超长上下文时才考虑 swap。

### 5.5 进阶：PD 分离（Prefill-Decode Disaggregation）

> 🔍 **延伸**：把 prefill 和 decode 放到**不同的实例/机器**上，通过高速网络传递 KV。

**动机**：prefill 是算力密集、decode 是带宽密集，混在一张卡上会互相干扰（prefill 拖慢 decode 的 ITL）。分离后可以各自用最优的并行策略和硬件配比。

**代价**：KV 需要跨节点传输，对网络带宽要求高（这是 vLLM、DeepSeek、Mooncake 等系统在解决的问题）。

---

## 6. 权衡：为什么"又快又高"很难

### 6.1 低延迟 vs 高吞吐

两者争夺同一套资源：算力、显存容量、显存带宽。

| 优化方向 | 倾向做法 | 代价 |
|---|---|---|
| 低 TTFT / 低 ITL | 小 batch、立即调度、少等待 | 硬件利用率低，总吞吐下降 |
| 高 tokens/s | 攒 batch、合并执行、提高并发 | 排队增加，TTFT 与尾延迟变差 |

在连续批处理中，新请求的 prefill 与老请求的 decode 还要共享同一批次：

- **优先新请求 prefill** → 新请求首 token 快，但存量请求 ITL 变长
- **优先存量 decode** → 输出稳定，但新请求 TTFT 上升

**所以线上不追求单一指标最大，而追求 SLO 下更高的 goodput。**

### 6.2 为什么 batch 变大后，单个请求的 TPOT 也变大

常见疑问：请求都塞进 batch 了，每个请求的算力不是没变吗？

**答**：小 batch 时近似如此；硬件满载后不是。请求不是各占一台设备，而是**共享**同一张卡的计算单元、HBM 带宽和缓存。

| Decode batch | 单步耗时 | 每请求 TPOT | 总吞吐 |
|---:|---:|---:|---:|
| 10 | 20 ms | 20 ms | 500 tok/s |
| 100 | 60 ms | 60 ms | 1,667 tok/s |

单步耗时增加的原因：

1. **总计算量增加**：大矩阵效率更高，但不是免费
2. **KV 读取量增加**：每个请求都要读自己的历史 KV，HBM 带宽有限
3. **并行资源饱和**：满载后新增工作带来排队、带宽竞争、同步开销

$$step\_time = f(\text{batch size}, \text{上下文长度}, \text{硬件资源})$$

> 🔍 **本质是 roofline 上的位置移动**：小 batch 在"带宽受限"区（加 batch 几乎免费，吞吐线性涨）；跨过脊点后进入"算力受限"区（加 batch 线性增加耗时，吞吐饱和）。这正好解释了 §2.3 的三种吞吐上限。

### 6.3 TPOT / ITL 受什么影响

"TPOT ≈ 单 step 耗时"是有用近似，但真实系统中它还反映**当前 step 混入了什么工作**：

1. **decode batch 中有多少请求** → 计算和 KV 带宽压力
2. **是否混入 prefill** → 新请求的 prompt 处理比单 token decode 重得多，会拖慢老请求
3. **是否发生抢占或重算** → 恢复时重算，表现为一次超长 ITL

$$\text{ITL} = \text{这位用户等下一个 token 时，系统完成当前混合调度批次的时间}$$

---

## 7. 优化技术地图

> 🔍 **延伸**：把散落的技术按"省什么资源"归类，方便定位。

| 技术 | 主要省 | 影响指标 | 备注 |
|---|---|---|---|
| **PagedAttention** | 显存（消除预留浪费） | 并发↑ → 吞吐↑ | vLLM 基石 |
| **GQA / MQA** | KV 显存 | 并发↑ | 改模型结构 |
| **MLA** | KV 显存 + 解压算力 | 并发↑、成本↓ | DeepSeek |
| **KV 量化**（FP8/INT4） | KV 显存、带宽 | 并发↑、TPOT↓ | 精度需验证 |
| **权重量化**（W4A16/FP8） | 权重显存、带宽 | TTFT↓、TPOT↓ | 移动 roofline 脊点 |
| **FlashAttention** | 激活显存、带宽 | TTFT↓ | 精确，非近似 |
| **Chunked prefill** | 激活峰值 | 保护 ITL | 牺牲自身 TTFT |
| **连续批处理** | 算力利用率 | 吞吐↑ | 现代引擎默认 |
| **Prefix caching** | KV 显存、prefill 算力 | TTFT↓、并发↑ | system prompt 复用 |
| **投机解码** | decode 步数 | TPOT↓ | 需 draft model |
| **PD 分离** | 资源冲突 | ITL↓、吞吐↑ | 需 KV 传输 |
| **TP / PP / EP** | 单卡显存 | 能跑更大模型 | 通信开销 |
| **CUDA Graph** | kernel launch 开销 | TPOT↓ | 占独立显存池 |
| **torch.compile** | 算子融合 | 带宽↓ | 编译时间成本 |

**一句话**：**显存类优化提升并发，带宽类优化降低 TPOT，算力类优化降低 TTFT。**

---

## 8. 常见误区 FAQ

**Q1：显存够大就能跑长上下文？**
不一定。权重能放下 ≠ 能跑。长上下文的**激活峰值**（尤其 logits 和 attention scores）和**KV 容量**才是瓶颈。4K prompt 的 prefill 峰值就能吃掉数 GB。

**Q2：batch 越大吞吐越高？**
到饱和点后收益递减直至为零，且 TPOT 变差。要看 roofline 上处于哪一段。

**Q3：TTFT 小就是好服务？**
不。可能只是系统在饿着跑（低并发）。要看 goodput。

**Q4：请求多了 KV cache 会把显存撑爆？**
不会。引擎预分配固定池 + 准入控制 + 抢占，满了是排队而不是 OOM。

**Q5：量化只省权重？**
KV Cache 可以单独量化（FP8 KV 很常见），激活也能量化（FP8 W8A8）。三者独立。

**Q6：TPOT 就是模型算一个 token 的时间？**
不是。它是"系统完成当前混合批次的时间"，包含排队、混批、抢占等系统级因素。

**Q7：MLA 只是把 KV 存小了？**
不止。矩阵吸收让它连"解压"都省了，是存储和计算的双重优化。

**Q8：chunked prefill 一定增加 TTFT？**
不一定。设备空闲时没有影响；只有并发负载下才有"chunk 间等待"。

---

## 9. 待办事项

> 完整版见 `TODO.md`

### [ ] 读懂 HuggingFace `transformers` 里 Llama 结构的源码

- 目标：`src/transformers/models/llama/modeling_llama.py`
- 要点：`LlamaConfig` 超参、`LlamaRMSNorm`、RoPE（`rotate_half` / `apply_rotary_pos_emb`）、SwiGLU 三矩阵、**GQA 的 `repeat_kv`**、**`DynamicCache.update` 的 prefill/decode 两条路径**、pre-norm 残差、4D mask 构造与 SDPA/eager/flash 分支、与 MLA 的对比
- 产出：阅读笔记 + 能凭记忆画出 forward 数据流

### [ ] 搞清楚推理引擎的显存管理是怎么做的

- **底层**：`memory_allocated` vs `memory_reserved`、caching allocator 分桶、碎片与 `expandable_segments`、`empty_cache()` 的真实作用
- **引擎**：`BlockAllocator` / `BlockSpaceManager` / `BlockTable` 职责、`block_size=16` 的权衡、逻辑块↔物理块映射、引用计数与 COW、启动 profiling 如何算 block 数、preemption 的 swap vs recompute
- **扩展**：Prefix caching / RadixAttention 树结构、CUDA Graph 私有显存池、TP/PP 切分、KV 量化后的 block 管理
- 产出：笔记 + "请求进来 → 分配 block → 用完回收"流程图 + 手算并发上限的例子

---

## 附录 A 术语表

| 术语 | 全称 | 含义 |
|---|---|---|
| MHA | Multi-Head Attention | 多头注意力 |
| MQA | Multi-Query Attention | 所有 head 共享一组 KV |
| GQA | Grouped-Query Attention | 分组共享 KV |
| MLA | Multi-head Latent Attention | 低秩压缩 + 解耦 RoPE |
| RoPE | Rotary Position Embedding | 旋转位置编码 |
| KV Cache | Key-Value Cache | 历史 token 的 K/V 缓存 |
| TTFT | Time To First Token | 首 token 延迟 |
| TPOT | Time Per Output Token | 平均每输出 token 耗时 |
| ITL | Inter-Token Latency | 相邻 token 间隔 |
| SLO | Service Level Objective | 服务等级目标 |
| Goodput | — | SLO 内的有效吞吐 |
| PagedAttention | — | 分块管理 KV 的注意力 |
| PD 分离 | Prefill-Decode Disaggregation | 预填充与解码分离部署 |
| COW | Copy-on-Write | 写时复制 |
| HBM | High Bandwidth Memory | 高带宽显存 |
| Roofline | — | 算力-带宽上界模型 |

## 附录 B 公式汇总

| 名称 | 公式 |
|---|---|
| 每 token KV（MHA/GQA） | $2 \cdot L \cdot h_{kv} \cdot d_h \cdot \text{bytes}$ |
| 每 token KV（MLA） | $L \cdot (d_c + d_R) \cdot \text{bytes}$ |
| 并发上限 | $\dfrac{\text{可用显存}}{\text{KV}_{token} \cdot \bar{S}}$ |
| TTFT | 排队 + prefill + 首次 decode |
| TPOT | $(\text{总时延} - TTFT)/(N_{out} - 1)$ |
| Goodput | 吞吐 × SLO 达标率 |
| Little's Law | $L = \lambda W$ |
| Roofline 脊点 | $F / \text{BW}$ |
| Prefill FLOPs | $\approx 2PS$ |
| Decode FLOPs/step | $\approx 2PB$ |
| 算力上限吞吐 | $\approx F/(2P)$ |
| 带宽上限吞吐（KV 主导） | $\approx \text{BW}/\text{KV}_{token}$ |

## 附录 C 参考资料

- Vaswani et al., *Attention Is All You Need*, 2017（MHA）
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019（MQA）
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models*, 2023
- DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, 2024（MLA、解耦 RoPE、矩阵吸收）
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023（vLLM）
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022
- Williams et al., *Roofline: An Insightful Visual Performance Model*, 2009

---

*文档整理时间：2026-10-28*
