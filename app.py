"""站台终端窄带组播补传演练网页。

流程：上传新图（≤32 KiB，切 4 KiB 块，至多 8 块）→ 登记 2~4 台终端首播
实收块号 → 逐轮规划补传（限帧数、限各终端异或解码次数，可注入某终端漏收
某帧）→ 各终端独立拼图，SHA-256 与新图相符才替换可下载的旧图。
"""
import hashlib
import io
import threading

from flask import (Flask, flash, redirect, render_template, request,
                   send_file, url_for)

from planner import (CHUNK_SIZE, MAX_CHUNKS, MAX_FRAME_LIMIT, apply_round,
                     plan_round, reassemble, split_chunks)
from progress import RoundIndex, build_round_table

app = Flask(__name__)
app.secret_key = "multicast-repair-drill-dev-key"

LOCK = threading.Lock()
MIN_TERMINALS = 2
MAX_TERMINALS = 4


def _fresh_state():
    return {"image": None, "terminals": [], "rounds": []}


STATE = _fresh_state()


def _mask_of(indices):
    m = 0
    for c in indices:
        m |= 1 << c
    return m


def _missing_list(mask, n_chunks):
    """缺块号，1 基，供页面显示。"""
    return [c + 1 for c in range(n_chunks) if not (mask >> c) & 1]


def _owned_list(mask, n_chunks):
    return [c + 1 for c in range(n_chunks) if (mask >> c) & 1]


def _fmt_frame(fr):
    kind, a, b = fr
    if kind == "single":
        return f"单块 #{a + 1}"
    return f"异或 #{a + 1}⊕#{b + 1}"


def _terminal_view(t, n_chunks):
    return {
        "name": t["name"],
        "owned": _owned_list(t["mask"], n_chunks),
        "missing": _missing_list(t["mask"], n_chunks),
        "completed": t["completed"],
        "total_decodes": t["total_decodes"],
    }


def _view_state():
    img = STATE["image"]
    n = img["n_chunks"] if img else 0
    return {
        "image": img,
        "terminals": [_terminal_view(t, n) for t in STATE["terminals"]],
        "rounds": STATE["rounds"],
        "chunk_size": CHUNK_SIZE,
        "max_chunks": MAX_CHUNKS,
        "max_frame_limit": MAX_FRAME_LIMIT,
        "min_terminals": MIN_TERMINALS,
        "max_terminals": MAX_TERMINALS,
        "term_rows": range(MAX_TERMINALS),
    }


@app.get("/")
def index():
    with LOCK:
        return render_template("index.html", **_view_state())


@app.post("/upload")
def upload():
    f = request.files.get("image")
    if not f or not f.filename:
        flash("未选择图片文件")
        return redirect(url_for("index"))
    data = f.read()
    try:
        chunks, size = split_chunks(data)
    except ValueError as e:
        flash(f"上传被拒：{e}")
        return redirect(url_for("index"))
    with LOCK:
        STATE.clear()
        STATE.update(_fresh_state())
        STATE["image"] = {
            "name": f.filename,
            "size": size,
            "sha256": hashlib.sha256(data).hexdigest(),
            "n_chunks": len(chunks),
            "chunks": chunks,
            "data": data,
        }
    flash(f"已载入新图：{f.filename}，{size} 字节，切为 {len(chunks)} 块。请登记终端首播实收块号。")
    return redirect(url_for("index"))


@app.post("/terminals")
def setup_terminals():
    with LOCK:
        img = STATE["image"]
        if not img:
            flash("请先上传新图")
            return redirect(url_for("index"))
        n = img["n_chunks"]
        terminals = []
        for i in range(MAX_TERMINALS):
            if not request.form.get(f"enable_{i}"):
                continue
            got = set()
            for v in request.form.getlist(f"recv_{i}"):
                c = int(v)
                if 1 <= c <= n:
                    got.add(c - 1)
            name = request.form.get(f"name_{i}", "").strip() or f"T{len(terminals) + 1}"
            mask = _mask_of(got)
            terminals.append({
                "name": name,
                "mask": mask,
                # 首播实收为进度重算起点，登记后不再改变
                "start_mask": mask,
                # 分层复合索引：维护与重算的工作量不随历史轮数线性增长
                "index": RoundIndex(n),
                "completed": False,
                "total_decodes": 0,
                "old": f"旧图占位：终端 {name} 尚未收齐新图。\n".encode(),
                "ext": ".txt",
            })
        if not (MIN_TERMINALS <= len(terminals) <= MAX_TERMINALS):
            flash(f"启用的终端数须为 {MIN_TERMINALS}~{MAX_TERMINALS} 台，当前 {len(terminals)} 台")
            return redirect(url_for("index"))
        STATE["terminals"] = terminals
        STATE["rounds"] = []
        # 首播即收齐的终端立即校验替换
        full = (1 << n) - 1
        for t in terminals:
            if t["mask"] == full:
                _try_finalize(t)
    flash(f"已登记 {len(terminals)} 台终端首播实收情况。")
    return redirect(url_for("index"))


def _try_finalize(t):
    """收齐后重组，SHA-256 与新图相符才替换可下载旧图。"""
    img = STATE["image"]
    data = reassemble(img["chunks"], img["size"])
    digest = hashlib.sha256(data).hexdigest()
    if digest == img["sha256"]:
        t["old"] = data
        t["ext"] = _guess_ext(img["name"])
        t["completed"] = True
        return True
    return False


def _revert_to_old(t):
    """更正后未收齐：恢复为旧图占位，撤回新图替换。"""
    t["completed"] = False
    t["old"] = f"旧图占位：终端 {t['name']} 尚未收齐新图。\n".encode()
    t["ext"] = ".txt"


def _guess_ext(filename):
    if "." in filename:
        return "." + filename.rsplit(".", 1)[1]
    return ".bin"


@app.post("/round")
def run_round():
    with LOCK:
        img = STATE["image"]
        terms = STATE["terminals"]
        if not img or not terms:
            flash("请先上传新图并登记终端")
            return redirect(url_for("index"))
        n = img["n_chunks"]
        try:
            frame_limit = int(request.form.get("frame_limit", ""))
        except ValueError:
            flash("帧数限额须为整数")
            return redirect(url_for("index"))
        if not (1 <= frame_limit <= MAX_FRAME_LIMIT):
            flash(f"帧数限额须在 1~{MAX_FRAME_LIMIT} 之间")
            return redirect(url_for("index"))
        quotas = []
        for i in range(len(terms)):
            try:
                q = int(request.form.get(f"quota_{i}", "0"))
            except ValueError:
                q = -1
            if not (0 <= q <= MAX_CHUNKS):
                flash(f"终端 {terms[i]['name']} 的解码额度须为 0~{MAX_CHUNKS} 的整数")
                return redirect(url_for("index"))
            quotas.append(q)
        fault = None
        ft = request.form.get("fault_terminal", "")
        ff = request.form.get("fault_frame", "").strip()
        if ft != "" and ff:
            try:
                fi, fn = int(ft), int(ff)
            except ValueError:
                fi, fn = -1, 0
            if 0 <= fi < len(terms) and fn >= 1:
                fault = (fi, fn)

        masks = [t["mask"] for t in terms]
        frames = plan_round(masks, quotas, frame_limit, n)
        if frames is None:
            flash("无方案：本轮限额内无法让各站收齐，已保留各站已收块，未发送任何帧。")
            return redirect(url_for("index"))
        if frames == []:
            flash("各站均已收齐，无需补传。")
            return redirect(url_for("index"))
        if fault and fault[1] > len(frames):
            fault = None  # 指定的漏收帧号超出实发帧数，忽略

        results = apply_round(masks, quotas, frames, fault)
        full = (1 << n) - 1
        rec_terminals = []
        for i, (t, r) in enumerate(zip(terms, results)):
            before = _missing_list(t["mask"], n)
            t["mask"] = r["mask"]
            t["total_decodes"] += r["decodes"]
            just_completed = False
            if t["mask"] == full and not t["completed"]:
                just_completed = _try_finalize(t)
            missed = r["missed_frame"]
            rec = {
                "name": t["name"],
                "before_missing": before,
                "after_missing": _missing_list(t["mask"], n),
                "received": r["received"],
                # 原回报与当前登记：更正只改 current_missed，原回报永久保留
                "original_missed": missed,
                "current_missed": missed,
                "decoded": [(no, c + 1) for no, c in r["decoded"]],
                "decodes": r["decodes"],
                "quota": quotas[i],
                "completed": t["completed"],
                "just_completed": just_completed,
            }
            rec_terminals.append(rec)
            # 该站本轮叶子转移表只回读本轮实发帧记录，追加进分层索引
            table = build_round_table(frames, quotas[i], missed, n)
            t["index"].append(table)
        STATE["rounds"].append({
            "no": len(STATE["rounds"]) + 1,
            "frame_limit": frame_limit,
            "frames": [_fmt_frame(fr) for fr in frames],
            # 不可变的实发原始记录，更正与重算只回读所改轮的这份记录
            "raw_frames": [tuple(fr) for fr in frames],
            "fault": (terms[fault[0]]["name"], fault[1]) if fault else None,
            "terminals": rec_terminals,
        })
        done = sum(1 for t in terms if t["completed"])
        flash(f"第 {len(STATE['rounds'])} 轮已发送 {len(frames)} 帧；{done}/{len(terms)} 台已收齐并校验替换。")
    return redirect(url_for("index"))


@app.post("/correct")
def correct_report():
    """漏帧回报更正：改记某站某轮为全收或漏收其中一帧。

    实发原始帧记录保留不变；只更新该站该轮的当前漏帧登记，并通过分层索引
    以首播实收为起点重算该站缺块与累计解码。其他站的进度与下载完全不变。
    任何校验失败都在改动状态前返回，收帧登记、进度和下载图保持原样。
    """
    with LOCK:
        img = STATE["image"]
        terms = STATE["terminals"]
        rounds = STATE["rounds"]
        # ---- 第一阶段：全部校验，不做任何状态修改 ----
        if not img or not terms or not rounds:
            flash("尚无已发送轮次可更正")
            return redirect(url_for("index"))
        n = img["n_chunks"]
        try:
            tid = int(request.form.get("terminal", ""))
            ridx = int(request.form.get("round", "")) - 1  # 轮号 1 基转 0 基
        except ValueError:
            flash("更正失败：终端或轮次选择无效，状态保持原样。")
            return redirect(url_for("index"))
        if not (0 <= tid < len(terms)) or not (0 <= ridx < len(rounds)):
            flash("更正失败：终端或轮次超出范围，状态保持原样。")
            return redirect(url_for("index"))
        mode = request.form.get("mode", "")
        if mode not in ("all", "miss"):
            flash("更正失败：须选择改记为全收或漏收一帧，状态保持原样。")
            return redirect(url_for("index"))
        round_rec = rounds[ridx]
        n_frames = len(round_rec["raw_frames"])
        if mode == "all":
            new_missed = None
        else:
            try:
                new_missed = int(request.form.get("miss_frame", ""))
            except ValueError:
                new_missed = -1
            if not (1 <= new_missed <= n_frames):
                flash(f"更正失败：漏收帧号须在 1~{n_frames} 之间，状态保持原样。")
                return redirect(url_for("index"))
        t = terms[tid]
        rec = round_rec["terminals"][tid]
        if rec["current_missed"] == new_missed:
            flash("更正失败：所选登记与当前登记相同，无需更正，状态保持原样。")
            return redirect(url_for("index"))

        # ---- 第二阶段：只回读所改轮的原始帧记录重建叶表，沿祖先链更新索引 ----
        table = build_round_table(
            round_rec["raw_frames"], rec["quota"], new_missed, n
        )
        t["index"].replace_leaf(ridx, table)
        new_mask, new_decodes = t["index"].current(t["start_mask"])

        # ---- 第三阶段：提交该站登记、进度与下载图 ----
        old_completed = t["completed"]
        rec["current_missed"] = new_missed
        t["mask"] = new_mask
        t["total_decodes"] = new_decodes
        full = (1 << n) - 1
        if new_mask == full:
            if not old_completed:
                # 收齐且通过 SHA-256 校验才供新图；校验不过则维持旧图
                _try_finalize(t)
        else:
            if old_completed:
                _revert_to_old(t)
        if new_mask == full and t["completed"]:
            outcome = "已收齐并通过 SHA-256 校验，恢复供新图。"
        elif new_mask == full:
            outcome = "块已收齐但 SHA-256 校验未通过，仍供旧图。"
        else:
            outcome = "尚未收齐，已恢复下载旧图。"
        label = "全收" if new_missed is None else f"漏收第 {new_missed} 帧"
        flash(
            f"已更正 {t['name']} 第 {round_rec['no']} 轮登记为{label}；"
            f"该站重算后缺块 {_missing_list(new_mask, n) or '无'}，"
            f"累计解码 {new_decodes} 次。{outcome}其他站进度与下载不变。"
        )
    return redirect(url_for("index"))


@app.post("/reset")
def reset():
    with LOCK:
        STATE.clear()
        STATE.update(_fresh_state())
    flash("演练已重置。")
    return redirect(url_for("index"))


@app.get("/terminal/<int:tid>/download")
def download(tid):
    with LOCK:
        if not (0 <= tid < len(STATE["terminals"])):
            flash("终端不存在")
            return redirect(url_for("index"))
        t = STATE["terminals"][tid]
        return send_file(
            io.BytesIO(t["old"]),
            as_attachment=True,
            download_name=f"{t['name']}_current{t['ext']}",
        )


@app.get("/image/download")
def download_new():
    with LOCK:
        img = STATE["image"]
        if not img:
            flash("尚未上传新图")
            return redirect(url_for("index"))
        return send_file(io.BytesIO(img["data"]), as_attachment=True,
                         download_name=img["name"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
