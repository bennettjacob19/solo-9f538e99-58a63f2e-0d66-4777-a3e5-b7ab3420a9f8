"""漏帧回报更正测试：分层索引正确性、复杂度约束、Flask 更正全流程。"""
import io
import math
import random

import pytest

import app as web
from planner import simulate_station
from progress import RoundIndex, build_round_table

ORIGINAL = (b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 34 + b"demo")  # 3 块


# ---------- 分层索引：与朴素逐轮重放对拍 ----------

def _naive_replay(start_mask, rounds):
    """朴素重放：逐轮回读全部原始帧记录，作为正确性基准。"""
    mask, decs = start_mask, 0
    for frames, quota, missed in rounds:
        r = simulate_station(mask, frames, quota, missed)
        mask, decs = r["mask"], decs + r["decodes"]
    return mask, decs


def test_index_matches_naive_replay_and_correction():
    rng = random.Random(20261125)
    n = 4
    idx = RoundIndex(n)
    start = 0b0011
    rounds = []
    for _ in range(37):
        frames = []
        for _ in range(rng.randint(1, 4)):
            if rng.random() < 0.5:
                frames.append(("single", rng.randrange(n), None))
            else:
                a, b = rng.sample(range(n), 2)
                frames.append(("xor", a, b))
        quota = rng.randint(0, 3)
        missed = rng.choice([None, 1, rng.randint(1, len(frames))])
        rounds.append((frames, quota, missed))
        idx.append(build_round_table(frames, quota, missed, n))

    assert idx.current(start) == _naive_replay(start, rounds)

    # 随机更正 10 个轮次：只重建所改轮叶表，结果必须与从头朴素重放一致
    for ridx in rng.sample(range(len(rounds)), 10):
        frames, quota, _ = rounds[ridx]
        new_missed = rng.choice([None] + list(range(1, len(frames) + 1)))
        rounds[ridx] = (frames, quota, new_missed)
        idx.replace_leaf(ridx, build_round_table(frames, quota, new_missed, n))
        assert idx.current(start) == _naive_replay(start, rounds)


def test_index_per_operation_work_is_logarithmic():
    """追加、更正、查询的复合次数均为 O(log 轮数)，不随历史轮数线性增长。"""
    n = 3
    frames = [("xor", 1, 2), ("single", 2, None)]
    R = 256
    idx = RoundIndex(n)
    worst_append = 0
    for _ in range(R):
        idx.compose_count = 0
        idx.append(build_round_table(frames, 1, None, n))
        worst_append = max(worst_append, idx.compose_count)
    assert worst_append <= math.ceil(math.log2(R)) + 1

    idx.compose_count = 0
    idx.replace_leaf(R // 2, build_round_table(frames, 1, 1, n))
    assert idx.compose_count <= math.ceil(math.log2(R)) + 1

    idx.compose_count = 0
    idx.current(0)
    assert idx.compose_count <= math.ceil(math.log2(R)) + 1


def test_replace_out_of_range_rejected():
    idx = RoundIndex(3)
    idx.append(build_round_table([("single", 0, None)], 0, None, 3))
    with pytest.raises(IndexError):
        idx.replace_leaf(5, build_round_table([("single", 0, None)], 0, None, 3))


# ---------- Flask 全流程 ----------

@pytest.fixture
def client():
    with web.LOCK:
        web.STATE.clear()
        web.STATE.update(web._fresh_state())
    web.app.config.update(TESTING=True)
    return web.app.test_client()


def _upload(client, data=ORIGINAL, name="new.png"):
    return client.post("/upload", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data", follow_redirects=True)


def _setup_two_terminals(client):
    # T1 首播收到 #1#2（缺#3），T2 收到 #1#3（缺#2）
    return client.post("/terminals", data={
        "enable_0": "on", "name_0": "T1", "recv_0": ["1", "2"],
        "enable_1": "on", "name_1": "T2", "recv_1": ["1", "3"],
    }, follow_redirects=True)


def _first_round_with_t1_miss(client):
    return client.post("/round", data={
        "frame_limit": "1", "quota_0": "1", "quota_1": "1",
        "fault_terminal": "0", "fault_frame": "1",
    }, follow_redirects=True)


def test_correct_to_all_received_completes_and_keeps_original_report(client):
    _upload(client)
    _setup_two_terminals(client)
    _first_round_with_t1_miss(client)
    with web.LOCK:
        raw_before = [tuple(fr) for fr in web.STATE["rounds"][0]["raw_frames"]]
        t1 = web.STATE["terminals"][0]
        assert t1["mask"] == 0b011 and not t1["completed"]
    assert client.get("/terminal/0/download").data != ORIGINAL

    r = client.post("/correct", data={"terminal": "0", "round": "1", "mode": "all"},
                    follow_redirects=True)
    assert "已更正".encode() in r.data
    # 页面同时展示原回报与当前登记
    assert "原回报".encode() in r.data and "当前".encode() in r.data
    assert "已更正".encode() in r.data
    with web.LOCK:
        round_rec = web.STATE["rounds"][0]
        rec = round_rec["terminals"][0]
        t1 = web.STATE["terminals"][0]
        assert rec["original_missed"] == 1       # 原回报保留
        assert rec["current_missed"] is None      # 当前登记为全收
        assert [tuple(fr) for fr in round_rec["raw_frames"]] == raw_before  # 实发记录不动
        assert t1["mask"] == 0b111 and t1["completed"]
        assert t1["total_decodes"] == 1           # 异或解回 #3，重算 1 次解码
    assert client.get("/terminal/0/download").data == ORIGINAL  # 过 SHA-256 才供新图


def test_correct_back_to_miss_reverts_old_image_and_recalculates(client):
    _upload(client)
    _setup_two_terminals(client)
    _first_round_with_t1_miss(client)
    client.post("/correct", data={"terminal": "0", "round": "1", "mode": "all"})
    assert client.get("/terminal/0/download").data == ORIGINAL

    client.post("/correct", data={"terminal": "0", "round": "1",
                                  "mode": "miss", "miss_frame": "1"})
    with web.LOCK:
        t1 = web.STATE["terminals"][0]
        rec = web.STATE["rounds"][0]["terminals"][0]
        assert rec["current_missed"] == 1
        assert t1["mask"] == 0b011 and not t1["completed"]
        assert t1["total_decodes"] == 0
    assert client.get("/terminal/0/download").data != ORIGINAL  # 恢复旧图


def test_correction_does_not_touch_other_stations(client):
    _upload(client)
    _setup_two_terminals(client)
    _first_round_with_t1_miss(client)
    with web.LOCK:
        t2_before = dict(web.STATE["terminals"][1])
    client.post("/correct", data={"terminal": "0", "round": "1", "mode": "all"})
    with web.LOCK:
        t2 = web.STATE["terminals"][1]
        assert t2["mask"] == t2_before["mask"]
        assert t2["completed"] == t2_before["completed"]
        assert t2["total_decodes"] == t2_before["total_decodes"]
        assert t2["old"] == t2_before["old"]


def test_failed_correction_keeps_everything(client):
    _upload(client)
    _setup_two_terminals(client)
    _first_round_with_t1_miss(client)
    with web.LOCK:
        snapshot = repr(web.STATE)

    # 帧号超出实发帧数
    r = client.post("/correct", data={"terminal": "0", "round": "1",
                                      "mode": "miss", "miss_frame": "9"},
                    follow_redirects=True)
    assert "更正失败".encode() in r.data
    # 与当前登记相同
    r = client.post("/correct", data={"terminal": "0", "round": "1",
                                      "mode": "miss", "miss_frame": "1"},
                    follow_redirects=True)
    assert "无需更正".encode() in r.data
    # 轮次越界、模式非法
    client.post("/correct", data={"terminal": "0", "round": "99", "mode": "all"})
    client.post("/correct", data={"terminal": "0", "round": "1", "mode": "bogus"})
    with web.LOCK:
        assert repr(web.STATE) == snapshot
    assert client.get("/terminal/0/download").data != ORIGINAL


def test_next_round_uses_corrected_missing_blocks(client):
    _upload(client)
    _setup_two_terminals(client)
    _first_round_with_t1_miss(client)
    # 更正为全收后再改回漏收：下轮应按更正后的缺块 #3 规划，单块即可
    client.post("/correct", data={"terminal": "0", "round": "1", "mode": "all"})
    client.post("/correct", data={"terminal": "0", "round": "1",
                                  "mode": "miss", "miss_frame": "1"})
    r = client.post("/round", data={"frame_limit": "2", "quota_0": "0", "quota_1": "0"},
                    follow_redirects=True)
    assert "已发送 1 帧".encode() in r.data
    with web.LOCK:
        assert web.STATE["rounds"][1]["frames"] == ["单块 #3"]
        assert web.STATE["terminals"][0]["completed"]
    assert client.get("/terminal/0/download").data == ORIGINAL


def test_multi_round_correction_cascades_through_later_rounds(client):
    """更正早期轮次后，后续轮次的连锁效果由索引自动重算。"""
    _upload(client)
    _setup_two_terminals(client)
    # 第 1 轮 T1 漏收异或帧（缺 #3）
    _first_round_with_t1_miss(client)
    # 第 2 轮补发单块 #3，T1 收齐
    client.post("/round", data={"frame_limit": "2", "quota_0": "0", "quota_1": "0"})
    with web.LOCK:
        t1 = web.STATE["terminals"][0]
        assert t1["mask"] == 0b111 and t1["completed"] and t1["total_decodes"] == 0
    # 更正第 1 轮为全收：异或解回 #3（1 次解码），第 2 轮的 #3 变成重复单块、无增益
    client.post("/correct", data={"terminal": "0", "round": "1", "mode": "all"})
    with web.LOCK:
        t1 = web.STATE["terminals"][0]
        assert t1["mask"] == 0b111 and t1["completed"]
        assert t1["total_decodes"] == 1  # 解码次数随登记变化重算
    assert client.get("/terminal/0/download").data == ORIGINAL


def test_correction_form_renders_with_sent_rounds(client):
    _upload(client)
    _setup_two_terminals(client)
    _first_round_with_t1_miss(client)
    page = client.get("/").data
    assert "漏帧回报更正".encode() in page
    assert 'name="terminal"'.encode() in page
    assert 'name="round"'.encode() in page
    assert 'name="mode"'.encode() in page
