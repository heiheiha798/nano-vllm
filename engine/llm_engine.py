import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from models.config import Config
from engine.sequence import Sequence
from engine.sequence import SamplingParams
from engine.scheduler import Scheduler
from engine.model_runner import ModelRunner

"""
这个文件是整个推理 engine 的最外层入口。

它负责做三件事：

1. 接收外部 prompt，并把它们包装成 Sequence
2. 驱动 scheduler 和 model runner 一步一步前进
3. 把内部的逐步执行过程，整理成对外可用的 generate 接口

这一层不负责具体 attention 计算，也不直接管理 block。
它更像一个 orchestration 层，用来把：

- request 接口
- schedule
- model execution
- 最终输出

串成一个完整闭环。
"""


GenerateOutput = dict[str, str | list[int]]


class LLMEngine:
    # LLMEngine 是对外暴露的主入口。
    # 它把“用户请求”翻译成“内部一步步执行的状态机”。

    def __init__(self, model, **kwargs):
        # 只接收 Config 中真正定义过的字段，避免外部随意传无关参数。
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # Sequence.block_size 在整个 engine 内要保持一致，
        # 所以初始化时用 config 统一覆盖。
        Sequence.block_size = config.kvcache_block_size

        self.model_runner = ModelRunner(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

        # eos_token_id 直到 tokenizer 初始化后才能拿到，
        # 所以在这里回写进 config。
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)

        # 退出时自动回收 GPU 相关状态。
        atexit.register(self.exit)

    def exit(self):
        # 这里不直接知道底层释放细节，统一交给 model_runner。
        self.model_runner.call("exit")
        del self.model_runner

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # 外部可以直接传文本，也可以直接传 token ids。
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # step 是整个系统最关键的抽象边界：
        # 它只推进“一步”，而不是一次性生成完整结果。
        seqs, is_prefill = self.scheduler.schedule()

        # 这里用正负号区分 throughput 类型：
        # - 正数：prefill 处理了多少个 token
        # - 负数：decode 处理了多少个 Sequence
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        # 只有 finished 的 Sequence 才会真正对外输出结果。
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[GenerateOutput]:
        # generate 是一个同步的高级接口：
        # 内部其实仍然是不断调用 step() 推进状态。
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()

            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)

            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)

        pbar.close()

        # 对外返回时，统一按 seq_id 排序，保证输出顺序稳定。
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs


class LLM(LLMEngine):
    pass
