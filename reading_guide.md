# Nano-vLLM Reading Guide

这份导读不是对代码逐行解释，而是给出一个更适合初学者的阅读顺序，并把每一阶段最值得思考的问题提前摆出来。建议你边读边在纸上画状态流转，不要只盯着代码细节。

## 阅读目标

读完这个目录，最好能回答下面几个核心问题：

- 一个 request 进入 engine 之后，什么时候处于 waiting，什么时候处于 running？
- paged KV cache 里的 block 是怎么分配、复用、追加、释放的？
- prefill 和 decode 为什么必须分开调度？
- 模型前向时，Q/K/V 是怎么和 block table、slot mapping、context lens 接起来的？
- CUDA Graph 在这里到底 capture 了什么，为什么它更适合 decode 而不是 prefill？

## 建议阅读顺序

建议严格按下面顺序读，不要一上来就钻进 `models/qwen3.py`。

### 1. `engine/sequence.py`

先看一个“请求”在 engine 里最小会携带哪些状态。

这一层你要重点看：

- `SamplingParams`
- `SequenceStatus`
- `Sequence`
- `num_cached_tokens`
- `num_scheduled_tokens`
- `block_table`

读这一层时要回答：

- 为什么 `Sequence` 既要保存 `token_ids`，又要保存 `last_token`？
- `num_prompt_tokens` 和 `num_completion_tokens` 分别服务于什么逻辑？
- `num_cached_tokens` 代表“已经生成的 token 数”，还是“已经写进 KV cache、因此不用重复 prefill 的 token 数”？
- `block_table` 为什么不直接存 K/V tensor，而是只存 block id？
- 为什么 `append_token()` 只管逻辑 token 流，不直接操作 KV cache？

读完这一层后，你应该先建立一个心智模型：

- `Sequence` 是“请求的逻辑状态”
- KV cache 不是直接挂在 `Sequence` 上，而是通过 `block_table` 间接关联

### 2. `engine/block_manager.py`

这一层是理解 paged KV cache 的关键。

这一层你要重点看：

- `Block`
- `free_block_ids`
- `used_block_ids`
- `hash_to_block_id`
- `allocate()`
- `deallocate()`
- `can_append()`
- `may_append()`

读这一层时要回答：

- 为什么 block manager 管的是 block id，而不是直接管一整块 GPU tensor？
- `allocate()` 里为什么要区分 cache hit 和 cache miss？
- `hash_to_block_id` 的作用是什么？它是在做 prefix cache 吗？
- 为什么只有“满块”才会被计算 hash，而不完整的最后一块不会？
- `may_append()` 为什么要区分三种情况：
  - 新 token 刚好让序列进入新块
  - 新 token 刚好填满最后一块
  - 最后一块还没满
- 为什么 block 的生命周期要由 `ref_count` 管，而不是简单地 append / pop？

建议你在这里手动画一个例子：

- block size = 4
- 序列 token 长度从 `4 -> 5 -> 6 -> 8 -> 9`
- 看 `block_table`、`last_block`、`hash` 是怎么变化的

如果这一层没看懂，后面 `scheduler` 和 `attention` 基本都会变成黑箱。

### 3. `engine/scheduler.py`

这一层回答的是：什么时候做 prefill，什么时候做 decode，以及在显存紧张时怎么抢资源。

这一层你要重点看：

- `waiting`
- `running`
- `schedule()`
- `preempt()`
- `postprocess()`

读这一层时要回答：

- 为什么 scheduler 必须显式区分 prefill 路径和 decode 路径？
- 为什么 prefill 阶段按“token budget”调度，而 decode 阶段几乎总是“每个序列 1 token”？
- `schedule()` 里为什么只允许“第一个序列”做 chunked prefill？
- `preempt()` 为什么不是简单暂停，而是直接 `deallocate()` 再放回 waiting？
- `postprocess()` 里为什么 prefill 完成后不一定立刻 append 新 token？
- 一个序列什么时候算 finished？是遇到 EOS 就结束，还是达到 `max_tokens` 也结束？

建议你在这里手动画两个队列：

- `waiting = [A, B]`
- `running = [C, D]`

然后模拟三轮：

1. 一轮 prefill
2. 一轮 decode
3. 一次因为 block 不够触发的 preemption

你要能说清楚：

- 哪个序列被挪到了哪里
- 哪些 block 被释放
- 哪些 token 会在下一轮重复 prefill

### 4. `engine/llm_engine.py`

这一层是整个系统的最外层控制入口。

这一层你要重点看：

- `LLMEngine.__init__`
- `add_request()`
- `step()`
- `generate()`

读这一层时要回答：

- 为什么 `add_request()` 只负责把 prompt 转成 `Sequence`，而不直接做模型前向？
- `step()` 为什么只推进“一步”，而不是一次性生成完？
- `step()` 返回的 `num_tokens` 为什么 prefill 用正数、decode 用负数？
- `generate()` 的主循环里，prefill throughput 和 decode throughput 为什么要分开算？
- `generate()` 为什么只在 sequence finished 时把结果写入 outputs？

读完这一层后，你应该能一句话概括：

- `LLMEngine` 负责把“外部请求接口”翻译成 “scheduler + model runner” 之间的一步步协作

### 5. `engine/model_runner.py`

这一层最重要，也最容易让人迷失在实现细节里。建议第一次读时只抓主线，不要抠每个 tensor 的 dtype 和 stride。

这一层你要重点看：

- `warmup_model()`
- `allocate_kv_cache()`
- `prepare_prefill()`
- `prepare_decode()`
- `run_model()`
- `capture_cudagraph()`

读这一层时要回答：

- 为什么先 warmup，再计算可用显存，再分配 KV cache？
- `kv_cache` 的形状为什么是  
  `2 x num_layers x num_blocks x block_size x num_kv_heads x head_dim`？
- `prepare_prefill()` 为什么要构造：
  - `cu_seqlens_q`
  - `cu_seqlens_k`
  - `max_seqlen_q`
  - `max_seqlen_k`
  - `slot_mapping`
  - `block_tables`
- `prepare_decode()` 为什么不再构造 varlen 的 `cu_seqlens_*`，而改成 `context_lens + block_tables`？
- `run_model()` 为什么只有 decode 小 batch 才走 graph，prefill 不走？
- `capture_cudagraph()` 里为什么预先 capture 多个 batch size，而不是只 capture 一个 batch size？

这里建议你重点画两张图：

1. prefill 图  
   一个 batch 内有多个变长序列时，`input_ids / positions / cu_seqlens_q / cu_seqlens_k` 是怎么组织的？

2. decode 图  
   每个序列只输入一个 `last_token` 时，`slot_mapping / context_lens / block_tables` 是怎么告诉 attention 去哪里读历史 KV 的？

### 6. `utils/context.py`

这一层代码很短，但它是 attention 能拿到调度上下文的桥。

读这一层时要回答：

- 为什么 `prepare_prefill()` / `prepare_decode()` 不把所有上下文显式作为函数参数一路传下去？
- `context` 里最关键的字段是哪几个？
- 为什么 prefill 和 decode 需要不同的一组上下文字段？

如果你没看这层，后面 `layers/attention.py` 会显得像“平白无故就知道 block table 在哪里”。

### 7. `layers/attention.py`

这一层是“调度信息”真正进入 kernel 的地方。

这一层你要重点看：

- `store_kvcache_kernel`
- `store_kvcache()`
- `Attention.forward()`
- `flash_attn_varlen_func`
- `flash_attn_with_kvcache`

读这一层时要回答：

- 为什么要先把新算出来的 K/V 写回 cache，再做 attention？
- `slot_mapping` 到底在描述什么？它是在说“第几个 token 对应 cache 的哪个 slot”吗？
- prefill 为什么走 `flash_attn_varlen_func`？
- decode 为什么走 `flash_attn_with_kvcache`？
- 什么情况下 prefill 会把 `k, v` 替换成 `k_cache, v_cache`？
- `block_tables` 在 prefix cache 和 decode 两种场景里分别扮演什么角色？

建议你在这里把两种模式用一句话区分清楚：

- prefill：这次要算多个 query token，所以更像“变长批处理 attention”
- decode：这次每个序列只算 1 个 query token，所以更像“拿 1 个 q 去查整段历史 KV cache”

### 8. `models/qwen3.py`

前面主线清楚之后，再回来看模型结构。

这一层你要重点看：

- `Qwen3Attention`
- `Qwen3DecoderLayer`
- `Qwen3Model`
- `Qwen3ForCausalLM`

读这一层时要回答：

- `qkv_proj` 之后，Q/K/V 的 shape 分别是什么？
- `q_norm` / `k_norm` 为什么是在 head 维度上做？
- `rotary_emb` 是在 attention 前的哪一步插进去的？
- `compute_logits()` 为什么被单独拆出来，而不是直接把 lm head 写进 `forward()`？
- 为什么 `ParallelLMHead` 在 prefill 时只取每个序列最后一个位置的 hidden states？

这里你不需要把 Qwen3 的每个参数都背下来。对这个 repo，更重要的是搞清楚：

- 模型层是如何适配 engine 这套调度接口的

### 9. `layers/linear.py`、`layers/embed_head.py`、`layers/layernorm.py`、`layers/rotary_embedding.py`、`layers/sampler.py`

这些可以作为补充阅读。

阅读目的不是背 API，而是看：

- engine 需要的最小算子集合到底有哪些
- 哪些算子只是“普通神经网络层”
- 哪些算子会显式依赖调度上下文

其中最值得额外看的是：

- `layers/embed_head.py`
  为什么 prefill 只取每个序列最后一个位置的 hidden state 去算 logits
- `layers/sampler.py`
  为什么 sampler 接在 engine 里，而不是放进模型里

### 10. `bench_eager.py` 和 `bench_graph.py`

最后再看 benchmark，不要一开始就看。

看 benchmark 时要回答：

- 这里的 benchmark 到底在比较什么，不在比较什么？
- 为什么统一成 `prompt = 10 tokens`、`decode = 100 tokens`、只看 decode TPS？
- `bench_graph.py` 里 HF graph 和 nano-vllm graph 的 graph capture 粒度有什么不同？
- 为什么这里 graph 的意义主要出现在 decode，而不是 prefill？

你应该带着一个很明确的判断去看 benchmark：

- 它不是在证明“谁绝对更快”
- 它是在帮助你观察 runtime overhead、kernel 选择、graph capture 这几个因素如何影响 decode

## 一条推荐的阅读主线

如果你只想抓主线，不想第一次就读太多文件，可以用下面这条最短路径：

1. `engine/sequence.py`
2. `engine/block_manager.py`
3. `engine/scheduler.py`
4. `engine/llm_engine.py`
5. `engine/model_runner.py`
6. `utils/context.py`
7. `layers/attention.py`
8. `models/qwen3.py`

这条路径读完之后，再回头补 benchmark 和其他 layers，效果会更好。

## 读代码时最值得反复追问的 10 个问题

如果你只想带着几个大问题去读，建议反复追这 10 个：

1. `Sequence` 维护的是“逻辑 token 状态”，还是“物理显存状态”？
2. `block_table` 为什么是 paged KV cache 的核心抽象？
3. prefix cache 到底复用的是什么，是 hidden states、K/V，还是 block 映射？
4. 为什么 prefill 和 decode 不能用同一套调度策略？
5. 为什么 decode 通常是“每个序列 1 个 query token”，但仍然要读完整历史 KV？
6. `slot_mapping` 到底描述的是“新 token 应该写入 cache 的哪里”，还是“历史 token 在哪里”？
7. 为什么 graph capture 更适合 decode，而不适合一般形式的 prefill？
8. `flash_attn_varlen_func` 和 `flash_attn_with_kvcache` 分别对应哪种 attention 形态？
9. `LLMEngine.step()` 为什么是整个系统最关键的抽象边界？
10. 如果以后要扩展 continuous batching、更多请求类型或更复杂的抢占策略，现在哪个文件最先需要改？

## 一个更有效的阅读习惯

建议你每读完一层，都先停下来，不要立刻往下翻。

你最好自己写出三句话：

1. 这一层的数据结构是什么？
2. 这一层对上一层暴露了什么接口？
3. 这一层替下一层准备了什么信息？

如果这三句话写不出来，说明你应该回去重读，而不是继续往下。
