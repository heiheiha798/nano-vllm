# Scheduler Guide

这份文档专门解释 [`scheduler.py`](/data/home/tianjianyang/code/aisys-map/experiments/nano-vllm/engine/scheduler.py)。

如果说：

- [`sequence.py`](/data/home/tianjianyang/code/aisys-map/experiments/nano-vllm/engine/sequence.py) 定义了一个 request 的逻辑状态
- [`block_manager.py`](/data/home/tianjianyang/code/aisys-map/experiments/nano-vllm/engine/block_manager.py) 管理 paged KV cache 的物理 block

那么 `scheduler.py` 负责回答的就是：

- 现在应该执行哪些 Sequence
- 当前这一轮应该走 prefill 还是 decode
- block 不够时，应该抢占谁
- 一轮模型执行结束后，Sequence 的状态该怎么更新

这正是之后阅读 vLLM scheduler、continuous batching、preemption 策略时最应该先建立的心智模型。

## 一句话理解 Scheduler

Scheduler 是一个“每次只推进一步”的调度器。

每调用一次 `schedule()`，它只做一件事：

- 要么挑出一批 Sequence 做 prefill
- 要么挑出一批 Sequence 做 decode

它不会一次性把一个 request 整个跑完，而是不断重复：

1. 选一批 Sequence
2. 执行一轮模型
3. 根据结果更新状态
4. 再进入下一轮

这种“一步一步推进”的结构，是 continuous batching 成立的前提。

## 先看三个核心对象

### 1. waiting

`waiting` 队列里放的是“已经进入系统，但当前还没真正执行”的 Sequence。

典型来源有两种：

- 一个新 request 刚进入系统
- 一个原本在 running 里的 request 被 preempt，释放 block 后重新回到 waiting

注意：

- waiting 不等于“完全没做过任何事情”
- 一个被 preempt 的 Sequence 也会回到 waiting

### 2. running

`running` 队列里放的是“已经拿到 block，并且当前仍然活跃”的 Sequence。

这些 Sequence 的共同特点是：

- 它们已经不再是刚进入系统的全新 request
- 它们随时可能在下一轮参与 decode
- 它们占用了 paged KV cache 的 block 资源

### 3. block_manager

Scheduler 自己不直接分配显存，但它会频繁询问和调用 `block_manager`：

- `can_allocate(seq)`
- `allocate(seq)`
- `can_append(seq)`
- `may_append(seq)`
- `deallocate(seq)`

所以可以把 Scheduler 理解成：

- 决策层：决定“谁该跑”
- BlockManager：资源层，决定“有没有空间让它跑”

## 为什么必须区分 prefill 和 decode

这是整个调度逻辑里最重要的一点。

### prefill 的特点

prefill 阶段处理的是“还没有进入有效 KV cache 的 token”。

对一个 Sequence 来说，prefill 可能要一次处理很多 token：

- 新 request 第一次进入系统时，通常要处理完整 prompt
- 被 preempt 之后，也可能要重新补一段 prefill

所以 prefill 的资源单位更像是：

- 一轮处理多少个 token

### decode 的特点

decode 阶段不同。

对一个活跃 Sequence 来说，每一轮 decode 只会新增 1 个 token。

也就是说：

- 一次 decode 调度，通常是“每个 Sequence 处理 1 个 token”
- 但这个 1 个 query token 会读取整段历史 KV cache

所以 decode 的资源单位更像是：

- 一轮处理多少个 Sequence

### 这就是为什么两条路径不能混成一套逻辑

prefill 更像：

- 按 token budget 装箱
- 看一轮能塞多少 token

decode 更像：

- 从 running 队列里挑活跃 request
- 每个 request 给 1 个 token 的执行机会

如果把这两者混成一个统一调度器，代码会更难读，也更难做 throughput accounting。

## `schedule()` 的整体结构

`schedule()` 的结构非常直接：

1. 先尝试 prefill
2. 如果这轮能调出至少一个 prefill Sequence，就直接返回
3. 否则再进入 decode 路径

也就是说，在这份实现里：

- prefill 优先级高于 decode

这不是唯一可能的设计，但它让调度逻辑更容易理解。

## prefill 路径详解

先看这一段主逻辑：

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]
    num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
    remaining = self.max_num_batched_tokens - num_batched_tokens
```

这里在做两件事：

- 看当前 waiting 队首的 Sequence 还剩多少 token 需要处理
- 看当前这轮 batch 还剩多少 token budget

### `num_tokens` 为什么是 `seq.num_tokens - seq.num_cached_tokens`

因为 prefill 的目标不是“把整个序列再算一遍”，而是：

- 只补那些还没有有效 KV cache 的 token

如果一个 Sequence 已经 cache 了前 100 个 token，现在有 120 个 token：

- 那么真正需要 prefill 的只剩 20 个

### 为什么还要 `max(..., 1)`

这是因为：

- 即使 prompt 已经完全 cache 了，decode 之前也至少要保留一个可推进的最小单位

你可以把它理解成一种防御性的最小推进逻辑。

### prefill 什么时候直接停止

这段代码是第一组停止条件：

```python
if remaining == 0 or (not seq.block_table and not self.block_manager.can_allocate(seq)):
    break
```

表示两种情况不再继续往 batch 里塞 prefill：

1. 这一轮 token budget 已经没了
2. 这是一个全新的 Sequence，但当前 block 不够给它分配

这里要注意：

- 对于全新 Sequence，prefill 前必须先拿到 `block_table`
- 如果 block 连预留都做不到，它就根本不能进入执行

### 为什么只允许“第一个 Sequence”做 chunked prefill

代码是：

```python
if remaining < num_tokens and scheduled_seqs:
    break
```

意思是：

- 如果当前 budget 不够覆盖这个 Sequence 的全部剩余 token
- 但前面已经塞过别的 Sequence 了
- 那就不再让当前这个 Sequence 也做 chunked prefill

这条规则的核心目的不是性能，而是降低调度复杂度。

因为如果一轮 batch 里有多个 Sequence 都被切碎：

- 谁补到哪里
- 哪些 token 已 cache
- 哪些 Sequence 该进 running

都会更难推理。

所以这份教学实现选择了一个更容易理解的约束：

- 最多只让 batch 里的第一个 Sequence 做 chunked prefill

### 什么时候从 waiting 进入 running

```python
if seq.num_scheduled_tokens == num_tokens:
    seq.status = SequenceStatus.RUNNING
    self.waiting.popleft()
    self.running.append(seq)
```

意思是：

- 如果这一轮把它剩余需要 prefill 的 token 全部覆盖掉了
- 那它就不再是“等待进入系统”的状态
- 而是正式进入 running

这说明：

- `running` 的含义不是“这一轮正在执行”
- 而是“这个 Sequence 已经进入活跃状态，之后可以参与 decode”

## decode 路径详解

如果这轮一个 prefill 都没调出来，就进入 decode：

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
```

decode 路径只看 `running` 队列，因为：

- 只有已经拥有 block 和有效上下文的 Sequence，才有资格继续 decode

### 为什么 decode 每轮只给 1 个 token

```python
seq.num_scheduled_tokens = 1
```

这是 decode 的基本语义：

- 每个活跃 Sequence 每轮只新增 1 个 token

这里最容易误解的点是：

- “只处理 1 个 token”不代表计算很小
- 因为 attention 仍然要读取整段历史 KV cache

所以 decode 的代价结构是：

- query 长度是 1
- key/value 长度是历史上下文长度

### `can_append()` 和 `may_append()` 在 decode 里分别干什么

`can_append(seq)` 回答的是：

- 如果这个 Sequence 这一轮再多 1 个 token，block 资源够不够

`may_append(seq)` 做的是：

- 在真正执行模型之前，提前把 block_table 结构准备好
- 确保这个新 token 生成后，有合法的 cache 写入位置

所以它们的关系是：

1. `can_append()` 先判断资源
2. `may_append()` 再更新 block 结构

### 为什么 block 不够时要 preempt

这一段是 decode 的关键：

```python
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())
    else:
        self.preempt(seq)
        break
```

意思是：

- 当前这个 Sequence 想继续 decode
- 但显存 block 不够了
- 那就必须有人让出 block

优先策略是：

1. 先抢占 `running` 队尾的别的 Sequence
2. 如果已经没有别的 Sequence 可抢，就连当前这个 Sequence 自己也得被抢占

这就是一个非常小型的“资源回收 + 公平推进”机制。

## `preempt()` 到底意味着什么

这是理解 vLLM 调度时最容易绕晕的一点。

这里的 `preempt()` 不是：

- 暂停计算并原地保留所有状态

这里的 `preempt()` 实际上是：

1. 把状态设回 `WAITING`
2. 调用 `block_manager.deallocate(seq)` 释放所有 block
3. 把它塞回 `waiting` 队列头部

也就是说，语义上更接近：

- “你先出去，等资源够了再回来重新补 cache”

这就是为什么文档里一直强调：

- 被 preempt 的 Sequence 之后可能要重新 prefill

## `postprocess()` 在做什么

`schedule()` 只决定“这轮谁来跑”。
真正一轮模型执行结束后，还要有人负责把结果写回 Sequence，这就是 `postprocess()`。

### prefill 之后为什么不一定 append 新 token

这一段最关键：

```python
if is_prefill:
    seq.num_cached_tokens = min(seq.num_cached_tokens + seq.num_scheduled_tokens, seq.num_tokens)
    if seq.num_cached_tokens < seq.num_tokens or seq.num_completion_tokens > 0:
        seq.num_scheduled_tokens = 0
        continue
```

这里表达的是：

- prefill 的首要目标是补齐 KV cache
- 不是每做完一轮 prefill 都应该立刻生成新 token

两种情况不能 append：

1. chunked prefill  
   prompt 还没 prefill 完

2. re-prefill after preemption  
   之前被抢占了，现在只是把 cache 补回来

只有当：

- prompt 已经完整 prefill
- 并且当前不是“在恢复现场”

才应该真正进入“生成新 token”的阶段。

### 什么时候真正 append 新 token

```python
seq.append_token(token_id)
seq.num_cached_tokens += 1
```

这说明一次真正的生成会带来两件事：

1. 逻辑 token 序列增加了一个 token
2. 这个新 token 也立刻拥有了对应的 cache

### 什么时候结束

有两个停止条件：

```python
if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
```

也就是：

1. 生成到 EOS，且没有忽略 EOS
2. completion token 数达到 `max_tokens`

结束后会：

- 设为 `FINISHED`
- 释放 block
- 从 `running` 移除

## 一个完整例子

下面用一个简化例子，把这份 scheduler 的工作流串一遍。

假设：

- `max_num_seqs = 2`
- `max_num_batched_tokens = 8`
- block 资源暂时够用
- 现在有两个 request：
  - A: prompt 长度 6
  - B: prompt 长度 5

### 第 1 轮

初始状态：

- `waiting = [A, B]`
- `running = []`

prefill 先看 A：

- A 还需要 6 个 token
- budget 还有 8
- A 可以完整塞入

再看 B：

- B 还需要 5 个 token
- budget 只剩 2
- 因为 A 已经在 `scheduled_seqs` 里了，不能让 B 也做 chunked prefill

所以第 1 轮只跑 A 的 prefill。

跑完后：

- A 进入 `running`
- B 还留在 `waiting`

状态变成：

- `waiting = [B]`
- `running = [A]`

### 第 2 轮

prefill 先看 B：

- B 还需要 5 个 token
- budget 够

于是第 2 轮跑 B 的 prefill。

跑完后：

- B 进入 `running`

状态变成：

- `waiting = []`
- `running = [A, B]`

### 第 3 轮

现在 waiting 空了，所以直接进入 decode。

decode 会从 `running` 里取出 Sequence：

- A 本轮 schedule 1 个 token
- B 本轮 schedule 1 个 token

如果 block 也够，这一轮就会同时给 A 和 B 各生成 1 个 token。

### 如果某一轮 block 不够

假设此时 A 想继续 decode，但 block 不够 append。

那么 scheduler 会尝试：

- 先抢占 `running` 队尾的 B

结果会变成：

- B 被 `deallocate()`
- B 回到 `waiting`
- A 继续拿到执行机会

之后如果 B 重新回来，它可能要重新补一段 prefill。

这就是为什么：

- preemption 和 paged KV cache 是强耦合的

## 对以后阅读 vLLM 最重要的启发

虽然这份实现是教学版，但它已经把最重要的思想浓缩出来了。

### 1. continuous batching 不是“把请求堆一起”

它的本质是：

- 每一轮只推进一小步
- 每一轮都重新做调度决策
- 新请求、老请求、被抢占请求都可能在下一轮重新混进来

### 2. prefill 和 decode 的资源形态不同

prefill 更像：

- 按 token 数装箱

decode 更像：

- 按活跃 Sequence 数装箱

这也是很多推理系统要把两者分别优化的根本原因。

### 3. preemption 的代价很真实

被 preempt 不是“零成本暂停”。

在这份实现里，它意味着：

- block 被释放
- cache 映射消失
- 之后可能要重新 prefill

所以一个成熟系统一定要非常谨慎地决定：

- 什么时候值得抢占
- 抢占谁最划算

### 4. scheduler 和 block manager 必须联动理解

只看 scheduler，你会觉得它只是在排队。

只看 block manager，你会觉得它只是在分配块。

但真正关键的是：

- scheduler 的每个决策都受 block 资源约束
- block 资源状态又会反过来影响 scheduler 的策略

这也是为什么阅读 vLLM 时，scheduler 和 paged KV cache 绝对不能拆开看。

## 建议你带着这几个问题回去再读代码

1. 为什么 `running` 不等于“这一轮正在运行”，而是“当前活跃、可继续 decode”的集合？
2. 为什么 chunked prefill 在这份实现里只允许第一个 Sequence 使用？
3. 为什么 preempt 要直接释放 block，而不是保留现场？
4. 为什么 prefill 完成后不一定立刻生成新 token？
5. `num_cached_tokens` 在 scheduler 里到底起了什么作用？
6. 如果以后要支持更复杂的 continuous batching，这份实现里哪几行最先需要改？
