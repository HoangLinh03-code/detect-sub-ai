#!/usr/bin/env python3
"""
Hard-coded Chinese subtitle bounding-box pipeline (detection only, no recognition).

  Stage 1  detect : decode -> crop bottom ROI -> batch text detection -> cache boxes (JSON)
  Stage 2  smooth : filter -> merge boxes per line -> track over time -> one stable box per subtitle segment
  Stage 3  render : draw boxes on the original video, encode with ffmpeg (audio kept)

Usage
  python sub_boxes.py input.mp4 -o out.mp4                     # all stages
  python sub_boxes.py input.mp4 -o out.mp4 --stages smooth,render --max-gap 6   # re-tune without re-running the model
  python sub_boxes.py input.mp4 -o out_cpu.mp4 --device cpu    # CPU-only benchmark on the same machine
"""
import argparse
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np


# ----------------------------------------------------------------------------- utils
def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # pip install imageio-ffmpeg  (bundles an ffmpeg binary)
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def probe(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    meta = dict(
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 25.0,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return meta


def system_info():
    info = {"os": platform.platform(), "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(), "opencv": cv2.__version__}
    try:
        if platform.system() == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=15).stdout.strip()
            info["cpu"] = out or platform.processor()
        else:
            info["cpu"] = platform.processor() or platform.machine()
    except Exception:
        info["cpu"] = platform.processor()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        if out:
            info["gpu"] = out.splitlines()[0]
    except Exception:
        pass
    for mod in ("paddle", "paddleocr", "onnxruntime"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    return info


def paths(a):
    stem = os.path.splitext(os.path.basename(a.video))[0]
    os.makedirs(a.work, exist_ok=True)
    return {
        "dets": os.path.join(a.work, f"{stem}.dets.json"),
        "segs": os.path.join(a.work, f"{stem}.segments.json"),
        "timing": os.path.join(a.work, f"{stem}.timing.json"),
    }


def load_json(p, default=None):
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


# ----------------------------------------------------------------------------- stage 1: detect
def _field(res, key):
    """PaddleOCR 3.x result objects are dict-like; fall back to .json['res'] if not."""
    try:
        return res[key]
    except Exception:
        return res.json["res"][key]


def cmd_detect(a, P, timing):
    from paddleocr import TextDetection  # imported lazily so smooth/render work without paddle

    meta = probe(a.video)
    W, H = meta["width"], meta["height"]
    y0 = int(H * a.roi_top)
    scale = min(1.0, a.det_width / W)

    kw = dict(model_name=a.model, limit_side_len=a.det_width, limit_type="max")
    for k, v in (("device", a.device), ("engine", a.engine), ("thresh", a.thresh),
                 ("box_thresh", a.box_thresh), ("unclip_ratio", a.unclip)):
        if v is not None:
            kw[k] = v

    t0 = time.perf_counter()
    model = TextDetection(**kw)
    dummy = np.zeros((max(64, int((H - y0) * scale)), int(W * scale), 3), np.uint8)
    list(model.predict([dummy], batch_size=1))  # warm-up (CUDA init, cudnn autotune)
    warmup_s = time.perf_counter() - t0
    print(f"[detect] model={a.model} device={a.device or 'auto'} warm-up {warmup_s:.1f}s "
          f"ROI y>={y0} ({(1 - a.roi_top) * 100:.0f}% bottom), det width {int(W * scale)}")

    q = queue.Queue(maxsize=a.batch * 4)
    st = {"decode_s": 0.0}

    def reader():
        cap = cv2.VideoCapture(a.video)
        while True:
            t = time.perf_counter()
            ok, fr = cap.read()
            if not ok:
                break
            roi = fr[y0:, :]
            if scale < 1.0:
                roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            roi = np.ascontiguousarray(roi)
            st["decode_s"] += time.perf_counter() - t
            q.put(roi)
        cap.release()
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()

    frames, batch, det_s = [], [], 0.0

    def flush():
        nonlocal det_s
        if not batch:
            return
        t = time.perf_counter()
        outs = list(model.predict(batch, batch_size=len(batch)))
        det_s += time.perf_counter() - t
        if len(outs) != len(batch):
            raise RuntimeError(f"predict returned {len(outs)} results for {len(batch)} images")
        for res in outs:
            boxes = []
            for poly, sc in zip(_field(res, "dt_polys"), _field(res, "dt_scores")):
                p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                x1, y1, x2, y2 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
                boxes.append([round(float(x1 / scale), 1), round(float(y1 / scale + y0), 1),
                              round(float(x2 / scale), 1), round(float(y2 / scale + y0), 1),
                              round(float(sc), 3)])
            frames.append(boxes)
        batch.clear()

    t_wall = time.perf_counter()
    while True:
        item = q.get()
        if item is None:
            break
        batch.append(item)
        if len(batch) >= a.batch:
            flush()
            if len(frames) % (a.batch * 50) == 0:
                el = time.perf_counter() - t_wall
                print(f"[detect] {len(frames)} frames  {len(frames) / el:.1f} fps")
    flush()
    wall = time.perf_counter() - t_wall

    meta["n_frames_read"] = len(frames)
    save_json(P["dets"], {"meta": meta, "params": vars(a), "frames": frames})
    timing["detect"] = {"wall_s": round(wall, 2), "decode_roi_thread_s": round(st["decode_s"], 2),
                        "model_s": round(det_s, 2), "warmup_s": round(warmup_s, 2),
                        "frames": len(frames), "batch": a.batch, "device": a.device or "auto",
                        "model": a.model, "det_width": int(W * scale)}
    print(f"[detect] done: {len(frames)} frames in {wall:.1f}s ({len(frames) / wall:.1f} fps)")


# ----------------------------------------------------------------------------- stage 2: smooth
def merge_frame(boxes, gap_factor):
    """Merge fragments that belong to the same text line (same row, small horizontal gap)."""
    if not boxes:
        return []
    lines = []
    for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        for ln in lines:
            ly1, ly2 = min(x[1] for x in ln), max(x[3] for x in ln)
            inter = min(ly2, b[3]) - max(ly1, b[1])
            if inter > 0.5 * min(ly2 - ly1, b[3] - b[1]):
                ln.append(b)
                break
        else:
            lines.append([b])
    out = []
    for ln in lines:
        ln.sort(key=lambda b: b[0])
        cur = list(ln[0])
        for b in ln[1:]:
            h = max(cur[3] - cur[1], b[3] - b[1])
            if b[0] - cur[2] <= gap_factor * h:
                cur = [min(cur[0], b[0]), min(cur[1], b[1]), max(cur[2], b[2]), max(cur[3], b[3]),
                       min(cur[4], b[4])]
            else:
                out.append(cur)
                cur = list(b)
        out.append(cur)
    return out


def dominant_band(frames, H, bin_frac=0.01):
    """Most common vertical position (as fraction of H) of detected lines = the subtitle row."""
    cys = [((b[1] + b[3]) / 2) / H for f in frames for b in f]
    if not cys:
        return None
    nb = int(1 / bin_frac)
    hist, _ = np.histogram(cys, bins=nb, range=(0, 1))
    hist = np.convolve(hist, np.ones(3), mode="same")  # small smoothing
    return (int(np.argmax(hist)) + 0.5) * bin_frac


def build_tracks(frames, W, a):
    tracks, active = [], []
    x_tol = a.x_tol * W
    for f, boxes in enumerate(frames):
        active = [t for t in active if f - t["last"] <= a.max_gap + 1]
        pairs = []
        for bi, b in enumerate(boxes):
            for ti, t in enumerate(active):
                r = t["ref"]
                h = r[3] - r[1]
                dcy = abs((b[1] + b[3]) / 2 - (r[1] + r[3]) / 2)
                dx1, dx2 = abs(b[0] - r[0]), abs(b[2] - r[2])
                if dcy <= 0.5 * h and dx1 <= x_tol and dx2 <= x_tol:
                    pairs.append((dcy + dx1 + dx2, bi, ti))
        used_b, used_t = set(), set()
        for _, bi, ti in sorted(pairs):
            if bi in used_b or ti in used_t:
                continue
            used_b.add(bi)
            used_t.add(ti)
            t = active[ti]
            t["obs"].append((f, boxes[bi][:4]))
            t["last"] = f
            t["ref"] = boxes[bi][:4]
        for bi, b in enumerate(boxes):
            if bi not in used_b:
                t = {"obs": [(f, b[:4])], "last": f, "ref": b[:4]}
                tracks.append(t)
                active.append(t)
    return tracks


def cmd_smooth(a, P, timing):
    d = load_json(P["dets"])
    if d is None:
        raise SystemExit(f"Missing {P['dets']} - run the detect stage first")
    meta, raw = d["meta"], d["frames"]
    W, H = meta["width"], meta["height"]
    t0 = time.perf_counter()

    # 1) per-box filter (confidence, text height relative to frame height)
    frames = [[b for b in f if b[4] >= a.min_score and a.min_h * H <= (b[3] - b[1]) <= a.max_h * H]
              for f in raw]
    # 2) merge fragments into text lines
    frames = [merge_frame(f, a.gap_factor) for f in frames]
    # 3) line-level filter: wide enough, horizontally centred
    frames = [[b for b in f if (b[2] - b[0]) >= a.min_aspect * (b[3] - b[1])
               and abs((b[0] + b[2]) / 2 - W / 2) <= a.center_tol * W] for f in frames]
    # 4) keep only the dominant subtitle row (drops signs, logos, clothing text)
    band = dominant_band(frames, H)
    if band is not None:
        frames = [[b for b in f if abs(((b[1] + b[3]) / 2) / H - band) <= a.band_tol] for f in frames]

    # 5) temporal tracking with hole-filling tolerance
    tracks = build_tracks(frames, W, a)
    segs = []
    for t in tracks:
        n_obs = len(t["obs"])
        start, end = t["obs"][0][0], t["obs"][-1][0]
        if n_obs < a.min_len or n_obs / (end - start + 1) < a.min_density:
            continue
        box = np.median(np.array([b for _, b in t["obs"]]), axis=0)
        segs.append({"start": start, "end": end, "box": box.tolist(), "n_obs": n_obs})
    segs.sort(key=lambda s: s["start"])

    # 6) snap: re-join adjacent segments whose boxes are almost identical (removes tiny jumps)
    snap = a.snap_tol * W
    merged = []
    for s in segs:
        for m in reversed(merged[-4:]):
            if s["start"] - m["end"] - 1 <= 2 * a.max_gap and \
               all(abs(s["box"][i] - m["box"][i]) <= snap for i in range(4)):
                m["end"] = max(m["end"], s["end"])
                m["n_obs"] += s["n_obs"]
                break
        else:
            merged.append(dict(s))
    # 7) pad and clip
    for s in merged:
        x1, y1, x2, y2 = s["box"]
        s["box"] = [max(0, round(x1 - a.pad)), max(0, round(y1 - a.pad)),
                    min(W - 1, round(x2 + a.pad)), min(H - 1, round(y2 + a.pad))]
        s["start_s"] = round(s["start"] / meta["fps"], 3)
        s["end_s"] = round((s["end"] + 1) / meta["fps"], 3)

    save_json(P["segs"], {"meta": meta, "band": band, "segments": merged})
    timing["smooth"] = {"wall_s": round(time.perf_counter() - t0, 3), "segments": len(merged)}
    covered = sum(s["end"] - s["start"] + 1 for s in merged)
    print(f"[smooth] {len(merged)} subtitle segments, boxes on {covered}/{len(raw)} frames "
          f"({100 * covered / max(1, len(raw)):.0f}%), subtitle row at {band and round(band * 100)}% of height")


# ----------------------------------------------------------------------------- stage 3: render
def segments_to_frames(segs, n):
    per = [[] for _ in range(n)]
    for s in segs:
        for f in range(s["start"], min(s["end"], n - 1) + 1):
            per[f].append(s["box"])
    for f in range(n):  # drop boxes mostly contained in another box (duplicates / fragments)
        bs = per[f]
        keep = []
        for i, b in enumerate(bs):
            area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
            dup = False
            for j, c in enumerate(bs):
                if i == j:
                    continue
                iw = min(b[2], c[2]) - max(b[0], c[0])
                ih = min(b[3], c[3]) - max(b[1], c[1])
                carea = max(1, (c[2] - c[0]) * (c[3] - c[1]))
                if iw > 0 and ih > 0 and iw * ih / area > 0.85 and (area < carea or (area == carea and i > j)):
                    dup = True
                    break
            if not dup:
                keep.append(b)
        per[f] = keep
    return per


def cmd_render(a, P, timing):
    sd = load_json(P["segs"])
    if sd is None:
        raise SystemExit(f"Missing {P['segs']} - run the smooth stage first")
    meta = probe(a.video)
    W, H, fps, n = meta["width"], meta["height"], meta["fps"], meta["n_frames"]
    per = segments_to_frames(sd["segments"], max(n, sd["meta"].get("n_frames_read", n)) + 1)
    out = a.output or os.path.splitext(a.video)[0] + "_boxed.mp4"
    color = tuple(int(x) for x in a.color.split(","))  # BGR
    thick = a.thickness or max(2, round(H / 360))

    ff = ffmpeg_exe()
    proc, vw = None, None
    if ff:
        v = (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19"] if a.nvenc
             else ["-c:v", "libx264", "-preset", a.preset, "-crf", str(a.crf)])
        cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
               "-r", f"{fps:.6f}", "-i", "-", "-i", a.video, "-map", "0:v:0", "-map", "1:a?",
               *v, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    else:
        print("[render] ffmpeg not found (pip install imageio-ffmpeg) -> falling back to OpenCV mp4v, no audio")
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    cap = cv2.VideoCapture(a.video)
    t0 = time.perf_counter()
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        for x1, y1, x2, y2 in per[i] if i < len(per) else []:
            cv2.rectangle(fr, (x1, y1), (x2, y2), color, thick)
        if proc:
            try:
                proc.stdin.write(fr.tobytes())
            except BrokenPipeError:
                raise SystemExit("ffmpeg closed the pipe - check the ffmpeg error above")
        else:
            vw.write(fr)
        i += 1
    cap.release()
    if proc:
        proc.stdin.close()
        proc.wait()
    else:
        vw.release()
    wall = time.perf_counter() - t0
    timing["render"] = {"wall_s": round(wall, 2), "frames": i, "encoder": "h264_nvenc" if a.nvenc else "libx264"}
    print(f"[render] {i} frames -> {out} in {wall:.1f}s ({i / wall:.1f} fps)")


# ----------------------------------------------------------------------------- main
def report(timing, meta):
    dur = meta["n_frames"] / meta["fps"]
    total = sum(timing.get(k, {}).get("wall_s", 0) for k in ("detect", "smooth", "render"))
    print("\n=== TIMING ===")
    print(f"video : {meta['width']}x{meta['height']} @ {meta['fps']:.2f} fps, {meta['n_frames']} frames, {dur:.1f} s")
    for k in ("detect", "smooth", "render"):
        if k in timing:
            print(f"{k:7s}: {timing[k]['wall_s']:8.2f} s")
    if "detect" in timing:
        d = timing["detect"]
        print(f"         (detect breakdown: model {d['model_s']} s, decode+ROI thread {d['decode_roi_thread_s']} s, "
              f"warm-up {d['warmup_s']} s excluded)")
    print(f"TOTAL  : {total:8.2f} s  -> {meta['n_frames'] / max(total, 1e-9):.1f} fps, "
          f"RTF {total / dur:.2f} (processing time / video duration)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("-o", "--output")
    ap.add_argument("--work", default="work")
    ap.add_argument("--stages", default="detect,smooth,render")
    g = ap.add_argument_group("detect")
    g.add_argument("--model", default="PP-OCRv5_mobile_det",
                   help="NOTE: TextDetection() defaults to PP-OCRv6_medium_det if not set")
    g.add_argument("--device", default=None, help="e.g. gpu:0 or cpu (default: auto)")
    g.add_argument("--engine", default=None, help="paddle_static (default) | onnxruntime")
    g.add_argument("--batch", type=int, default=8)
    g.add_argument("--roi-top", type=float, default=0.55, help="ROI starts at this fraction of height (0.55 = bottom 45%%)")
    g.add_argument("--det-width", type=int, default=960, help="ROI is downscaled to this width before detection")
    g.add_argument("--thresh", type=float)
    g.add_argument("--box-thresh", type=float)
    g.add_argument("--unclip", type=float)
    s = ap.add_argument_group("smooth")
    s.add_argument("--min-score", type=float, default=0.5)
    s.add_argument("--min-h", type=float, default=0.02, help="min text height / frame height")
    s.add_argument("--max-h", type=float, default=0.12, help="max text height / frame height")
    s.add_argument("--gap-factor", type=float, default=1.5, help="merge boxes on a line if gap <= factor * text height")
    s.add_argument("--min-aspect", type=float, default=1.5, help="min line width / height")
    s.add_argument("--center-tol", type=float, default=0.25, help="max |line centre - frame centre| / width")
    s.add_argument("--band-tol", type=float, default=0.08, help="tolerance around dominant subtitle row (fraction of height)")
    s.add_argument("--x-tol", type=float, default=0.03, help="same-sentence tolerance for line edges (fraction of width)")
    s.add_argument("--max-gap", type=int, default=5, help="frames a track may vanish and still be continued")
    s.add_argument("--min-len", type=int, default=3, help="min observed frames per segment")
    s.add_argument("--min-density", type=float, default=0.3, help="min observed/length ratio per segment")
    s.add_argument("--snap-tol", type=float, default=0.006, help="re-join adjacent segments with edges within this fraction of width")
    s.add_argument("--pad", type=int, default=4)
    r = ap.add_argument_group("render")
    r.add_argument("--color", default="0,255,0", help="BGR, e.g. 0,255,0 = green, 0,0,255 = red")
    r.add_argument("--thickness", type=int, default=0)
    r.add_argument("--nvenc", action="store_true", help="use NVIDIA h264_nvenc encoder (faster on RTX)")
    r.add_argument("--crf", type=int, default=18)
    r.add_argument("--preset", default="veryfast")
    a = ap.parse_args()

    P = paths(a)
    timing = load_json(P["timing"], {}) or {}
    stages = [x.strip() for x in a.stages.split(",")]
    if "detect" in stages:
        timing.pop("smooth", None)
        timing.pop("render", None)
    timing["system"] = system_info()
    meta = probe(a.video)
    timing["video"] = meta
    for st in ("detect", "smooth", "render"):
        if st in stages:
            {"detect": cmd_detect, "smooth": cmd_smooth, "render": cmd_render}[st](a, P, timing)
            save_json(P["timing"], timing)
    report(timing, meta)
    print(json.dumps(timing["system"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()