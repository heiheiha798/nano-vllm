from collections import deque
from models.config import Config
from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager

"""
这个文件负责 request 的调度，也就是决定：

1. 哪些 Sequence 现在可以执行
2. 当前这一轮执行 prefill 还是 decode
3. block 不够时，哪个 Sequence 会被 preempt

这里最重要的不是模型计算，而是状态流转：

- waiting: 已进入系统，但当前还没执行
- running: 已经拿到 block，正在参与后续 prefill / decode
- finished: 已满足停止条件

这一层把“逻辑 request 状态”和“物理 block 资源”连接了起来。
"""


class Scheduler:
    # Scheduler 维护两个核心队列：
    #
    # - waiting: 等待被调度的 Sequence
    # - running: 已经处于活动状态的 Sequence

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        # 所有队列都空了，说明整个 engine 当前没有未完成 request。
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # 新 request 统一先进入 waiting。
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # 返回值：
        #
        # - scheduled_seqs: 这一轮真正会被执行的 Sequence
        # - is_prefill: 这一轮是不是 prefill
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens

            # token budget 已经没有了，当前轮次就不能再塞更多 prefill。
            if remaining == 0:
                break

            # 对于一个全新 Sequence，先探测有多少前缀 block 可以直接复用。
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 对于 chunked prefill，真正还需要补算的是“未 cache 的尾部”。
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # 这里只允许“第一个 Sequence”做 chunked prefill。
            # 原因是如果一个 batch 里多个 Sequence 都被切碎，调度和 accounting 会更复杂。
            if remaining < num_tokens and scheduled_seqs:
                break

            if not seq.block_table:
                # 新 Sequence 第一次进入执行前，要先拿到自己的 block_table。
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            # 如果本轮把它剩余的 prefill token 全部覆盖掉，
            # 就可以把它从 waiting 挪到 running。
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)

            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()

            # decode 阶段每个 Sequence 一轮只追加 1 个 token，
            # 但在真正执行前，要先确认 block 资源是否足够。
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 优先抢占队尾 Sequence，让当前这个 Sequence 尽量继续推进。
                    self.preempt(self.running.pop())
                else:
                    # 如果连当前 Sequence 自己都保不住，就只能抢占它自己。
                    self.preempt(seq)
                    break
            else:
                # decode 阶段，每个 Sequence 每轮只 schedule 1 个 token。
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        assert scheduled_seqs

        # 这一步把本轮被执行的 Sequence 重新放回 running，
        # 保证它们仍然保持“活跃”状态。
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # preempt 的语义不是“暂停并保留现场”，
        # 而是释放它当前持有的 block，让它回到 waiting 以后重新竞争资源。
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        # postprocess 负责把这轮 model 执行结果写回 Sequence 状态。
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            # prefill 的职责首先是把已有 token 的 KV cache 补齐。
            # 只有当前序列已经完整 cache 后，才能真正生成下一个 token。
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            # 只有真正进入“生成新 token”阶段时，才会 append_token。
            seq.append_token(token_id)

            # 两种停止条件：
            # 1. 生成了 EOS，且没有忽略 EOS
            # 2. completion token 数已经达到 max_tokens
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
