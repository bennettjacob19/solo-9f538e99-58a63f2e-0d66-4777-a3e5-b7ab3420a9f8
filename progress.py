"""进度重算索引：把每轮抽成一个"块掩码转移表"，用分层段树维护。

为什么不随历史轮数线性增长：
- 一轮对某站的作用只取决于该站进入本轮时的块掩码（解码计数每轮从零起算），
  因此可表示为一张至多 ``1 << n_chunks`` 项的转移表 table[entry_mask] =
  (exit_mask, round_decodes)。建表只回读该轮自己的原始帧记录，与历史轮数无关。
- 各轮转移表按轮序做函数复合，索引按层存放：seg[k][j] 覆盖轮次区间
  [j*2^k, (j+1)*2^k)，由两个半段复合而成。
- 追加一轮只新增 O(log 轮数) 个段；更正一轮只重建该轮叶表及其祖先链，
  也是 O(log 轮数) 次定长表复合；查当前状态同样是 O(log 轮数) 次复合。
- 单次表复合是 2^块数（≤256）项的定长循环，与历史轮数无关。
"""
import math

from planner import simulate_station


def identity_table(size):
    """恒等转移：原样通过，不产生解码。"""
    return [(m, 0) for m in range(size)]


def build_round_table(frames, quota, missed_frame, n_chunks):
    """只回读本轮原始帧记录，建出该站本轮的完整转移表。"""
    size = 1 << n_chunks
    table = [None] * size
    for entry in range(size):
        r = simulate_station(entry, frames, quota, missed_frame)
        table[entry] = (r["mask"], r["decodes"])
    return table


def compose(outer, inner):
    """先过 inner 再过 outer 的复合转移表（轮序：inner 在前，outer 在后）。"""
    out = [None] * len(inner)
    for m, (mid_mask, dec_in) in enumerate(inner):
        out_mask, dec_out = outer[mid_mask]
        out[m] = (out_mask, dec_in + dec_out)
    return out


class RoundIndex:
    """单站全部轮次的分层复合索引（seg[k][j] 覆盖轮次 [j*2^k, (j+1)*2^k)）。"""

    def __init__(self, n_chunks):
        self.n_chunks = n_chunks
        self.size = 1 << n_chunks
        self.seg = [[]]            # seg[0] 为各轮叶表，按轮序排列
        self.round_count = 0
        self.compose_count = 0    # 统计复合次数，供测试验证复杂度

    def _compose(self, outer, inner):
        self.compose_count += 1
        return compose(outer, inner)

    def append(self, table):
        """追加一轮转移表，新增段数 O(log 轮数)，叶表永久保留。"""
        r = self.round_count
        self.seg[0].append(table)
        k = 1
        # 叶子 r（0 基）凑满长度 2^k 的段：(r+1) 是 2^k 的倍数
        while (r + 1) % (1 << k) == 0:
            j = r >> k
            left = self.seg[k - 1][2 * j]
            right = self.seg[k - 1][2 * j + 1]
            if k == len(self.seg):
                self.seg.append([])
            self.seg[k].append(self._compose(right, left))
            k += 1
        self.round_count += 1

    def replace_leaf(self, round_idx, table):
        """替换第 round_idx 轮（0 基）叶表并只重建其祖先链。

        只有叶表重建回读了所改轮的原始帧记录；祖先全部是定长表复合，
        复合次数不超过层数 O(log 轮数)。
        """
        if not (0 <= round_idx < self.round_count):
            raise IndexError("round index out of range")
        self.seg[0][round_idx] = table
        r = self.round_count
        k = 1
        while (1 << k) <= r:
            j = round_idx >> k
            if (j + 1) * (1 << k) > r:
                break  # 包含该轮的这一层段尚未成形，无需也无法重建
            left = self.seg[k - 1][2 * j]
            right = self.seg[k - 1][2 * j + 1]
            self.seg[k][j] = self._compose(right, left)
            k += 1

    def current(self, start_mask):
        """从首播实收掩码出发，复合全部轮次，返回 (当前掩码, 累计解码次数)。"""
        mask, decs = start_mask, 0
        pos = 0
        r = self.round_count
        max_level = len(self.seg) - 1
        while pos < r:
            k = min(max_level, int(math.log2(r - pos)))
            j = pos >> k
            table = self.seg[k][j]
            mask, add_decs = table[mask]
            decs += add_decs
            self.compose_count += 1
            pos += 1 << k
        return mask, decs
