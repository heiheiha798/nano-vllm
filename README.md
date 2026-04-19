# Nano-vLLM

这个目录不是为了复刻一个完整可发布的推理框架，而是为了把 `vLLM / paged KV cache / scheduler / CUDA Graph / continuous batching` 这些概念收敛到一个足够小、可以顺着读下去的实现里。

它的定位是教学和参考，不是生产级 engine：

- 重点是把推理引擎的关键逻辑串起来。
- 重点是帮助理解 request 如何进入 scheduler，KV block 如何分配，decode 如何调度。
- 不重点追求接口完整性、异常处理、可配置性和极限性能。

## 建议阅读顺序

如果你第一次看这个目录，建议按下面顺序读：

1. `engine/sequence.py`
2. `engine/block_manager.py`
3. `engine/scheduler.py`
4. `engine/llm_engine.py`
5. `engine/model_runner.py`
6. `layers/attention.py`
7. `models/qwen3.py`

这样先建立“请求和 KV cache 怎么流动”的心智模型，再回头看具体模型层实现，会更顺。

## 目录结构

- `engine/`
  推理引擎主流程。包含 sequence、scheduler、block manager、model runner。
- `layers/`
  模型里真正会参与前向计算的基础层，尤其是 attention、RoPE、RMSNorm、linear、sampler。
- `models/`
  模型结构和配置，目前主要是 Qwen3。
- `utils/`
  上下文传递和权重加载等辅助逻辑。
- `bench_eager.py`
  decode-only eager benchmark。
- `bench_graph.py`
  decode-only CUDA Graph benchmark。

## 最小使用方式

当前目录是 repo 内部实验代码，不是独立 pip 包。最简单的运行方式是从仓库根目录设置 `PYTHONPATH`：

```bash
PYTHONPATH=experiments/nano-vllm conda run -n aisys python experiments/nano-vllm/bench_eager.py
```

如果只是想最小跑通一次生成，也可以直接在脚本里调用：

```python
from engine.llm_engine import LLM
from engine.sequence import SamplingParams

llm = LLM("/data/pretrained_models/Qwen3-0.6B", enforce_eager=True)
sampling_params = SamplingParams(temperature=0.8, max_tokens=32)
outputs = llm.generate(["Hello"], sampling_params, use_tqdm=False)
print(outputs[0]["text"])
```

## Benchmark 脚本说明

这两个 benchmark 都是教学用 benchmark，不是严肃论文式 benchmark。

- `bench_eager.py`
  对比 `HF eager no FA2`、`HF eager + FA2`、`nano-vllm eager + FA2`
- `bench_graph.py`
  对比 `HF + SDPA + graph`、`nano-vllm + FA2 + graph`

两个脚本都统一为：

- prompt 固定 `10` tokens
- sampled decode 固定 `100` tokens
- 只统计 decode 阶段 TPS

这样做的目的不是给出通用结论，而是帮助理解：

- eager 和 graph 的差异主要体现在哪里
- HF 原生路径和一个更接近 serving engine 的路径差异在哪里
- 当 decode 每步计算量很小时，runtime overhead 为什么会变得显著

## 这份实现刻意保留的简化

为了可读性，这里故意没有把事情做得很“工业化”：

- 只覆盖当前学习路径需要的模型和功能
- 没有做完整的配置系统和命令行封装
- 没有做完整的错误处理
- 没有覆盖真正复杂的多请求服务场景
- benchmark 也只覆盖了单请求、小 batch 的教学场景

所以读这个目录时，最好的预期不是“拿来直接做服务”，而是：

- 先看懂 paged KV cache 和 block table 是怎么工作的
- 先看懂 scheduler 为什么要区分 prefill 和 decode
- 先看懂 graph capture 为什么更适合固定形状的 decode 子问题

## 为什么直接锁死单卡

这里直接假设单卡即可，不再保留 tensor parallel / distributed 的教学复杂度：

- 当前学习模型是 `Qwen3-0.6B`
- 这个量级本来就适合单卡学习
- 多卡会把注意力从 `KV cache / scheduler / graph / block table` 转移到分布式通信

所以这里的选择是主动收敛复杂度，不是假装支持完整 serving 场景。

## 目前最值得关注的文件

- `engine/block_manager.py`
  看 block 是怎么 allocate / append / deallocate 的。
- `engine/scheduler.py`
  看 waiting / running 两个队列怎么切换，以及 prefill / decode 是怎么分开的。
- `engine/model_runner.py`
  看模型前向、KV cache、graph capture 怎么接起来。
- `layers/attention.py`
  看 flash attention 和 KV cache 写回是怎么接上的。

## 运行前提

- 当前环境默认使用本机已有 CUDA 和 PyTorch
- 模型路径默认写的是 `/data/pretrained_models/Qwen3-0.6B`
- 需要可用的 `flash_attn`

如果 `flash_attn` 不可用，这个目录里的核心 attention 路径就跑不通。
