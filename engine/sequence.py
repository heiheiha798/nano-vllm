from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count

"""
这个文件只负责“逻辑 request 状态”，不直接管理物理 KV cache tensor。
你可以把 Sequence 理解为：

1. 它知道当前这个 request 有哪些 token
2. 它知道这些 token 里哪些已经被 cache / schedule 处理过
3. 它通过 block_table 间接关联 paged KV cache

这里最重要的区分是：

- token_ids: 逻辑上的 token 序列
- num_cached_tokens: 已经进入 KV cache、因此下次不需要重复 prefill 的 token 数
- num_scheduled_tokens: 当前这一轮 schedule 准备送去 model runner 的 token 数
- block_table: 这个序列映射到了哪些 KV cache block

换句话说，Sequence 不负责“算 attention”，它负责给 scheduler 和 block manager
提供最小但足够的状态。
"""


@dataclass(slots=True)
class SamplingParams:
    # temperature 控制采样的随机性。
    # 这里故意不允许 greedy，是为了让教学脚本始终走同一条 sampling 路径。
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"


class SequenceStatus(Enum):
    # WAITING:
    # request 已进入 engine，但当前还没有被执行。
    WAITING = auto()
    # RUNNING:
    # request 已经拿到 block，并且会参与后续 prefill / decode。
    RUNNING = auto()
    # FINISHED:
    # request 已经满足停止条件，比如 EOS 或 max_tokens。
    FINISHED = auto()


class Sequence:
    # block_size 是所有 Sequence 共享的全局分块大小。
    # 它决定逻辑 token 序列如何被切成一个个 KV cache block。
    block_size = 256
    # counter 用来为每个 request 生成唯一的 seq_id。
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        # seq_id 是一个 request 在 engine 内部的稳定身份标识。
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING

        # 这里保存的是“逻辑 token 流”，不是物理显存布局。
        self.token_ids = copy(token_ids)

        # 单独缓存 last_token，是因为 decode 阶段通常只需要最新一个 token 作为输入。
        self.last_token = token_ids[-1]

        self.num_tokens = len(self.token_ids)

        # num_prompt_tokens 标记原始 prompt 的边界。
        # 整个序列可以理解为：
        # [prompt tokens | completion tokens]
        self.num_prompt_tokens = len(token_ids)

        # 它不是“已经生成了多少 token”，
        # 而是“已经拥有有效 KV cache、下次不用重复 prefill 的 token 数”。
        self.num_cached_tokens = 0

        # num_scheduled_tokens 表示当前这一轮 schedule 选中了多少 token，
        # 这些 token 会立刻送去执行。
        self.num_scheduled_tokens = 0

        # block_table 把逻辑 token 序列映射到物理 KV cache block。
        # 它是 paged KV cache 的核心索引，不直接存 K/V，只存 block id。
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        # 这是给 scheduler / engine 使用的便捷状态视图。
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        # completion tokens 指原始 prompt 之后新生成的部分。
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        # 当前逻辑 token 序列一共需要多少个逻辑 block。
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        # 当前最后一个逻辑 block 里有多少个 token。
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        # 返回第 i 个逻辑 block。
        # 这里切的是逻辑 token 分块，不是物理 cache block tensor。
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        # 这里只更新逻辑 token 状态。
        # 真正把 K/V 写进 cache 的动作不在这里，而在 attention / model runner 里。
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        # 序列被序列化时，尽量只保留恢复所需的最小状态。
        # 如果 completion 还没开始，或者还有 token 需要重新 prefill，
        # 就必须保留完整 token_ids。
        # 否则，在“已经完全 cache 化的 decode 状态”下，
        # 只保留 last_token 就够了。
        last_state = (
            self.token_ids
            if self.num_completion_tokens == 0 or self.num_cached_tokens < self.num_tokens
            else self.last_token
        )
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            last_state,
        )

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # 在紧凑的 decode-only 状态下，这里故意不恢复完整 token_ids。
            self.token_ids = []
            self.last_token = last_state
