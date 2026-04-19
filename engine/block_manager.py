from collections import deque
import xxhash
import numpy as np
from engine.sequence import Sequence

"""
这个文件负责管理 paged KV cache 的 block 生命周期。

你可以把 BlockManager 理解为一个“物理块分配器”：

1. Sequence 只知道自己逻辑上有哪些 token、有哪些 block id
2. BlockManager 负责决定这些 block id 如何分配、复用、追加、释放
3. prefix cache 也是在这一层通过 hash 机制实现的

这里最重要的几个概念是：

- free_block_ids: 当前还没被占用的 block
- used_block_ids: 当前正在被使用的 block
- hash_to_block_id: 用于 prefix cache 的哈希索引
- ref_count: 一个 block 当前被多少个 Sequence 共享

换句话说，这一层管理的是“物理 block 状态”，不是 request 的调度顺序。
"""


class Block:
    # Block 是一个物理 KV cache block 的元信息。
    # 它不直接存 GPU tensor，只存管理信息。

    def __init__(self, block_id):
        self.block_id = block_id
        # ref_count 表示当前有多少个 Sequence 在引用这个 block。
        self.ref_count = 0
        # hash 只对“满块”有意义，用于 prefix cache 命中判断。
        self.hash = -1
        # token_ids 只作为校验和缓存复用判断使用。
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        # 当一个满块可以参与 prefix cache 时，记录它的 hash 和 token 内容。
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        # 一个刚分配出去的新 block，天然已经被一个 Sequence 持有。
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    # BlockManager 管理的是 block id 到 block 状态的映射。
    # 真正的 K/V tensor 在 model runner 里，这里只维护分配关系。

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        # hash_to_block_id 用于 prefix cache：
        # 如果一个满块 token 序列的 hash 已经出现过，就可以尝试直接复用。
        self.hash_to_block_id: dict[int, int] = dict()

        # free_block_ids / used_block_ids 一起描述当前 block 池状态。
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 这里的 hash 不是只描述“当前 block 的 token 内容”，
        # 而是描述“从序列开头到当前 block 为止的整个前缀”。
        #
        # 具体做法是：
        #
        # 1. 如果 prefix != -1，先把前一段前缀的 hash 喂进去
        # 2. 再把当前 block 的 token_ids 喂进去
        # 3. 最后得到一个新的 int hash
        #
        # 这样做的意义是：
        #
        # - 如果两个 Sequence 当前 block 的 token 一样，但前面历史不同，
        #   那么它们的 hash 也不同
        # - 只有“前面的前缀一样 + 当前 block 内容也一样”时，
        #   才会命中同一个 hash
        #
        # 这正是 prefix cache 需要的语义。
        #
        # 举例：
        #
        # block_size = 4
        # Sequence A:
        # [10, 11, 12, 13] [20, 21, 22, 23]
        #
        # 第 0 块的 hash 近似表示为：
        # hash([10, 11, 12, 13])
        #
        # 第 1 块的 hash 近似表示为：
        # hash(hash([10, 11, 12, 13]) + [20, 21, 22, 23])
        #
        # 所以第 1 块的命中，要求的不只是“第二块一样”，
        # 而是“前面的完整前缀也一样”。
        #
        # 这里把 token_ids 转成原始 bytes 再做 hash，
        # 是为了得到稳定、紧凑的二进制表示。
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        # 从 free 列表中拿出一个 block，变成 used 状态。
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> Block:
        # 一个 block 的 ref_count 降到 0 后，才能真正回到 free 列表。
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        # 一个全新 Sequence 进入系统时，需要先为它的所有逻辑 block
        # 预留出足够的物理 block。
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        # allocate 只用于“还没有 block_table 的新 Sequence”。
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)

            # 只有满块才会参与 prefix cache。
            # 不完整的最后一块内容还不稳定，不值得进入 hash 索引。
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)

            # 即使 hash 命中，也要再比较 token_ids，避免极端哈希冲突。
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            if cache_miss:
                # 一旦某一块 miss，后面所有块都不能继续沿着这个前缀复用，
                # 因为前缀链已经断了。
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 命中 prefix cache，说明这一整块已经拥有有效 KV cache，
                # 不需要重复 prefill。
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 某些 block 虽然当前不在 used 集合里，但仍然可以通过
                    # hash 索引重新拿出来复用。
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        # 释放一个 Sequence 持有的所有 block 引用。
        # 注意这里只是减少 ref_count，只有 ref_count 归零才真正回收。
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        # 被抢占或结束后，这个 Sequence 不再拥有任何有效 cache 映射。
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # decode 阶段每追加一个 token，不一定都需要新 block。
        #
        # 只有当 len(seq) % block_size == 1 时，
        # 才说明刚刚进入了一个新的逻辑块，需要额外申请一个物理 block。
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        # 这个函数不直接追加 token，它只提前维护 block_table 结构，
        # 确保这次 decode 结束后，新的 token 有合法位置可写。
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        if len(seq) % self.block_size == 1:
            # 新 token 将进入一个全新的逻辑块。
            # 这意味着上一块已经是满块，因此应该已经拥有有效 hash。
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 新 token 会刚好把最后一块填满。
            # 这时最后一块终于可以计算 hash，纳入 prefix cache。
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 最后一块还没满，不需要额外操作。
            assert last_block.hash == -1
