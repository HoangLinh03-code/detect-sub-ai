#!/usr/bin/env python3
"""
DEBUG render script — run this to test improvements before modifying sub_boxes.py.

Changes vs. the original:
  1. Multi-row subtitle support: allows 2 vertical bands (main + wrap-around line)
     and merges adjacent rows into one tall bounding box when they overlap in time.
  2. Box info overlay: shows box coordinates (x1,y1)-(x2,y2) and W×H above each box.
  3. Timing overlay: shows subtitle segment start/end times (e.g. "01:23.5 - 01:25.8").
  4. Sub display bar: a semi-transparent bar below each box shows "sub: <detected text>"
     using PaddleOCR recognition.

Usage:
  python debug_render.py "AI Engineer test.mp4" -o "AI Debug.mp4"
  python debug_render.py "AI Engineer test.mp4" -o "AI Debug.mp4" --no-ocr   # skip OCR, show coords only
  python debug_render.py "AI Engineer test.mp4" -o "AI Debug.mp4" --stages smooth,render   # re-smooth only
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
from PIL import Image, ImageDraw, ImageFont


# ----------------------------------------------------------------------------- CJK font
def _find_cjk_font(size=20):
    """Find a CJK-capable TrueType font on the system."""
    import glob
    candidates = []
    # Windows fonts
    win_fonts = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    for name in ("msyh.ttc", "msyhbd.ttc", "simsun.ttc", "simhei.ttf", "STKAITI.TTF"):
        p = os.path.join(win_fonts, name)
        if os.path.exists(p):
            candidates.append(p)
    # Linux / Mac
    for pattern in ("/usr/share/fonts/**/Noto*CJK*.ttc",
                    "/usr/share/fonts/**/wqy*.ttc",
                    "/System/Library/Fonts/PingFang.ttc",
                    "/System/Library/Fonts/STHeiti*.ttc"):
        candidates.extend(glob.glob(pattern, recursive=True))
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()

_CJK_FONT_CACHE = {}
def get_cjk_font(size=20):
    if size not in _CJK_FONT_CACHE:
        _CJK_FONT_CACHE[size] = _find_cjk_font(size)
    return _CJK_FONT_CACHE[size]


def put_cjk_text(frame, text, pos, font_size, color_rgb, bg_color_rgb=None, bg_alpha=0.6, padding=4):
    """Draw Unicode text (including CJK) on an OpenCV frame using PIL."""
    font = get_cjk_font(font_size)
    # Measure text
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = pos
    # Draw background
    if bg_color_rgb is not None:
        overlay = frame.copy()
        bg_bgr = (bg_color_rgb[2], bg_color_rgb[1], bg_color_rgb[0])
        cv2.rectangle(overlay,
                      (x - padding, y - padding),
                      (x + tw + padding, y + th + padding),
                      bg_bgr, -1)
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    # Convert frame region to PIL, draw text, copy back
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((x, y), text, font=font, fill=color_rgb)
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return th + 2 * padding


# ============================================================================
# Utilities (copied from sub_boxes.py, unchanged)
# ============================================================================
def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
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


def load_json(p, default=None):
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def paths(a):
    stem = os.path.splitext(os.path.basename(a.video))[0]
    os.makedirs(a.work, exist_ok=True)
    return {
        "dets": os.path.join(a.work, f"{stem}.dets.json"),
        "segs": os.path.join(a.work, f"{stem}.segments_debug.json"),  # separate file for debug
    }


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


# ============================================================================
# Stage 1: detect (from sub_boxes.py)
# ============================================================================
def _field(res, key):
    """PaddleOCR 3.x result objects are dict-like; fall back to .json['res'] if not."""
    try:
        return res[key]
    except Exception:
        return res.json["res"][key]


def cmd_detect(a, P, timing):
    from paddleocr import TextDetection

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
    list(model.predict([dummy], batch_size=1))  # warm-up
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


# ============================================================================
# Stage 2: IMPROVED smooth — supports multi-line (wrapped) subtitles
# ============================================================================
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


def merge_multiline_boxes(boxes, H, max_row_gap_frac=0.06):
    """
    Merge vertically stacked subtitle lines into one bounding box.
    If two boxes are within max_row_gap_frac * H vertically and both horizontally centred,
    combine them into a single tall box.
    """
    if len(boxes) <= 1:
        return boxes

    max_gap = max_row_gap_frac * H
    # Sort by vertical center
    boxes_sorted = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    merged = [list(boxes_sorted[0])]

    for b in boxes_sorted[1:]:
        prev = merged[-1]
        # vertical gap between bottom of prev and top of current
        vgap = b[1] - prev[3]
        if vgap <= max_gap and vgap >= -(prev[3] - prev[1]) * 0.5:
            # Merge: extend the box
            merged[-1] = [min(prev[0], b[0]), min(prev[1], b[1]),
                          max(prev[2], b[2]), max(prev[3], b[3]),
                          min(prev[4], b[4])]
        else:
            merged.append(list(b))

    return merged


def filter_with_companion_rescue(boxes, W, H, min_aspect, max_row_gap_frac=0.06):
    """
    Filter boxes by min_aspect, but rescue short lines that are companions
    to a nearby strong (wide) line — these are multi-line subtitle wraps.

    A "weak" box (aspect < min_aspect) is kept if there exists a "strong" box
    in the same frame that is vertically adjacent (within max_row_gap_frac * H)
    and horizontally aligned (centers within 40% of the strong box's width).
    """
    if not boxes:
        return []

    max_vgap = max_row_gap_frac * H
    strong = []
    weak = []

    for b in boxes:
        bw = b[2] - b[0]
        bh = b[3] - b[1]
        if bh > 0 and bw >= min_aspect * bh:
            strong.append(b)
        else:
            weak.append(b)

    if not strong:
        return []  # no strong reference → can't rescue, drop all weak

    # Rescue weak boxes that are vertically adjacent to a strong box
    rescued = []
    for wb in weak:
        wb_cx = (wb[0] + wb[2]) / 2
        wb_h = wb[3] - wb[1]
        for sb in strong:
            sb_cx = (sb[0] + sb[2]) / 2
            sb_w = sb[2] - sb[0]
            sb_h = sb[3] - sb[1]
            # Vertical gap: positive = no overlap, negative = overlap
            vgap = max(wb[1] - sb[3], sb[1] - wb[3])
            if vgap > max_vgap:
                continue  # too far apart vertically
            min_h = min(wb_h, sb_h)
            if vgap < -0.5 * min_h:
                continue  # too much overlap = same line, handled by merge_frame
            # Horizontal alignment: weak box center within strong box's span
            if abs(wb_cx - sb_cx) <= 0.4 * sb_w:
                rescued.append(wb)
                break

    return strong + rescued


def dominant_bands(frames, H, bin_frac=0.01, n_bands=2):
    """Find up to n_bands dominant vertical positions (subtitle rows)."""
    cys = [((b[1] + b[3]) / 2) / H for f in frames for b in f]
    if not cys:
        return []
    nb = int(1 / bin_frac)
    hist, _ = np.histogram(cys, bins=nb, range=(0, 1))
    hist_smooth = np.convolve(hist, np.ones(3), mode="same")
    bands = []
    for _ in range(n_bands):
        peak = int(np.argmax(hist_smooth))
        if hist_smooth[peak] < 3:
            break
        bands.append((peak + 0.5) * bin_frac)
        # zero out neighbourhood to find next peak
        lo, hi = max(0, peak - 8), min(nb, peak + 9)
        hist_smooth[lo:hi] = 0
    return sorted(bands)


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


def cmd_smooth_debug(a, P):
    d = load_json(P["dets"])
    if d is None:
        raise SystemExit(f"Missing {P['dets']} — run the detect stage first (use sub_boxes.py)")
    meta, raw = d["meta"], d["frames"]
    W, H = meta["width"], meta["height"]
    t0 = time.perf_counter()

    # 1) per-box filter (confidence, text height relative to frame height)
    frames = [[b for b in f if b[4] >= a.min_score and a.min_h * H <= (b[3] - b[1]) <= a.max_h * H]
              for f in raw]

    # 2) merge fragments into text lines
    frames = [merge_frame(f, a.gap_factor) for f in frames]

    # 3) center filter only (defer aspect to companion rescue)
    frames = [[b for b in f if abs((b[0] + b[2]) / 2 - W / 2) <= a.center_tol * W] for f in frames]

    # 4) companion-aware aspect filter: keep wide boxes + rescue short companion lines
    frames = [filter_with_companion_rescue(f, W, H, a.min_aspect, max_row_gap_frac=0.06)
              for f in frames]

    # 4b) merge vertically stacked rows into single multi-line boxes
    frames = [merge_multiline_boxes(f, H, max_row_gap_frac=0.06) for f in frames]

    # 4c) dominant subtitle bands (computed after merge for accurate positions)
    bands = dominant_bands(frames, H, n_bands=2)
    if bands:
        band_tol = a.band_tol
        frames = [[b for b in f if any(abs(((b[1] + b[3]) / 2) / H - band) <= band_tol
                                       for band in bands)]
                  for f in frames]

    # 4d) final noise filter: minimum absolute width
    min_w_px = a.min_width_px if a.min_width_px > 0 else max(80, int(0.11 * W))
    frames = [[b for b in f if (b[2] - b[0]) >= min_w_px] for f in frames]

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

    # 6) snap: re-join adjacent segments
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

    save_json(P["segs"], {"meta": meta, "bands": bands, "segments": merged})
    covered = sum(s["end"] - s["start"] + 1 for s in merged)
    wall = time.perf_counter() - t0
    print(f"[smooth-debug] {len(merged)} subtitle segments, boxes on {covered}/{len(raw)} frames "
          f"({100 * covered / max(1, len(raw)):.0f}%), bands at {[round(b*100) for b in bands]}% of height")
    print(f"[smooth-debug] done in {wall:.3f}s")
    return merged


# ============================================================================
# OCR: recognise text inside each segment's bounding box (optional)
# ============================================================================
def recognise_segments(video_path, segments, meta):
    """Run PaddleOCR recognition on the middle frame of each segment (PaddleOCR 3.x API)."""
    try:
        from paddleocr import OCR
    except ImportError:
        try:
            from paddleocr import PaddleOCR as OCR
        except ImportError:
            print("[ocr] PaddleOCR not available — skipping text recognition")
            return {}

    print(f"[ocr] Recognising text in {len(segments)} segments...")
    try:
        ocr = OCR(lang="ch")
    except Exception:
        try:
            ocr = OCR()
        except Exception as e:
            print(f"[ocr] Failed to init OCR: {e}")
            return {}

    def _ocr_roi(roi):
        """Returns (text, avg_score). avg_score is 0 if unavailable."""
        try:
            results = list(ocr.predict(roi))
            lines, scores = [], []
            for res in results:
                r = None
                try:
                    r = res if isinstance(res, dict) else res.json["res"]
                except Exception:
                    r = None
                if isinstance(r, dict):
                    if "rec_texts" in r:
                        lines.extend(r["rec_texts"])
                        scores.extend(r.get("rec_scores", []) or [])
                    elif isinstance(r.get("res"), list):
                        for item in r["res"]:
                            if isinstance(item, dict) and "rec_text" in item:
                                lines.append(item["rec_text"])
                                if "rec_score" in item:
                                    scores.append(item["rec_score"])
            text = " ".join(lines) if lines else ""
            avg_score = sum(scores) / len(scores) if scores else 0.0
            return text, avg_score
        except Exception:
            return "", 0.0

    # IMPORTANT: the previous version read exactly ONE frame per segment (the
    # middle frame) via cap.set(cv2.CAP_PROP_POS_FRAMES, ...). Two problems
    # with that:
    #   1. cap.set() seeking is not frame-accurate on most H.264/VFR mp4 —
    #      the frame you actually get back can drift from the one you asked
    #      for (worse the more seeks you've already done), so OCR could read
    #      a neighbouring subtitle's text.
    #   2. Even with the *correct* frame, that single frame can happen to be
    #      mid-transition (fade in/out, motion blur) and simply contain no
    #      readable text — that's why e.g. segment #26 above came back empty
    #      even though it's a perfectly normal subtitle.
    #
    # Fix for both: decode the video sequentially exactly once (frame N is
    # guaranteed to be frame N), and for each segment sample several frames
    # spread across its duration instead of just the middle one. Whichever
    # sampled frame yields the longest / most confident recognised text for
    # that segment wins.
    N_SAMPLES = 5  # frames sampled per segment (raise if you still see gaps)

    candidates = []  # candidates[seg_idx] = sorted list of frame indices to try
    for seg in segments:
        s, e = seg["start"], seg["end"]
        length = e - s + 1
        if length <= 1 or N_SAMPLES <= 1:
            wanted = [(s + e) // 2]
        else:
            k = min(N_SAMPLES, length)
            wanted = sorted({s + round(i * (length - 1) / (k - 1)) for i in range(k)})
        candidates.append(wanted)

    frame_to_segs = {}
    for idx, flist in enumerate(candidates):
        for f in flist:
            frame_to_segs.setdefault(f, []).append(idx)

    cap = cv2.VideoCapture(video_path)
    per_seg_results = {idx: [] for idx in range(len(segments))}  # idx -> [(text, score), ...]
    remaining = len(frame_to_segs)
    i = 0
    while remaining > 0:
        ok, frame = cap.read()
        if not ok:
            break
        idxs = frame_to_segs.get(i)
        if idxs:
            for idx in idxs:
                x1, y1, x2, y2 = segments[idx]["box"]
                roi = frame[y1:y2, x1:x2]
                if roi.size:
                    text, score = _ocr_roi(roi)
                    if text:
                        per_seg_results[idx].append((text, score))
            remaining -= 1
        i += 1
    cap.release()

    texts = {}
    for idx in range(len(segments)):
        results = per_seg_results[idx]
        if not results:
            texts[idx] = ""
        else:
            # Prefer the longest recognised string (most complete read of the
            # line); break ties by OCR confidence.
            results.sort(key=lambda t: (len(t[0]), t[1]), reverse=True)
            texts[idx] = results[0][0]

    print(f"[ocr] Done — recognised {sum(1 for v in texts.values() if v)} / {len(segments)} segments "
          f"(sampled up to {N_SAMPLES} frames/segment)")
    return texts


# ============================================================================
# Stage 3: IMPROVED render — with overlays
# ============================================================================
def format_time(seconds):
    """Format seconds as MM:SS.d"""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


def put_text_with_bg(frame, text, pos, font_scale, color, thickness,
                     bg_color=(0, 0, 0), bg_alpha=0.6, padding=4):
    """Draw text with a semi-transparent background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # Background rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding),
                  bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    # Text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return th + baseline + 2 * padding


def segments_to_frames(segs, n):
    """Map each segment to its active frames, return list of (frame_index -> [(box, seg_index)])."""
    per = [[] for _ in range(n)]
    for si, s in enumerate(segs):
        for f in range(s["start"], min(s["end"], n - 1) + 1):
            per[f].append((s["box"], si))
    # Deduplicate: drop boxes mostly contained in another box
    for f in range(n):
        items = per[f]
        keep = []
        for i, (b, si) in enumerate(items):
            area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
            dup = False
            for j, (c, _) in enumerate(items):
                if i == j:
                    continue
                iw = min(b[2], c[2]) - max(b[0], c[0])
                ih = min(b[3], c[3]) - max(b[1], c[1])
                carea = max(1, (c[2] - c[0]) * (c[3] - c[1]))
                if iw > 0 and ih > 0 and iw * ih / area > 0.85 and (area < carea or (area == carea and i > j)):
                    dup = True
                    break
            if not dup:
                keep.append((b, si))
        per[f] = keep
    return per


def cmd_render_debug(a, P, segments, seg_texts):
    meta = probe(a.video)
    W, H, fps, n = meta["width"], meta["height"], meta["fps"], meta["n_frames"]
    per = segments_to_frames(segments, max(n, len(segments) and segments[-1]["end"] + 2) + 1)
    out = a.output or os.path.splitext(a.video)[0] + "_debug.mp4"
    color = tuple(int(x) for x in a.color.split(","))  # BGR
    thick = a.thickness or max(2, round(H / 360))

    # Scales for overlay text
    info_scale = max(0.35, H / 2800)
    sub_scale = max(0.45, H / 2200)
    info_thick = max(1, round(H / 900))

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
        print("[render] ffmpeg not found — falling back to OpenCV mp4v, no audio")
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    cap = cv2.VideoCapture(a.video)
    t0 = time.perf_counter()
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break

        items = per[i] if i < len(per) else []
        current_time = i / fps

        for box, si in items:
            x1, y1, x2, y2 = box
            seg = segments[si]

            # --- 1) Draw bounding box ---
            cv2.rectangle(fr, (x1, y1), (x2, y2), color, thick)

            # --- 2) Box info overlay (above box) ---
            bw, bh = x2 - x1, y2 - y1
            info_text = f"[{x1},{y1}]-[{x2},{y2}] {bw}x{bh}"
            put_text_with_bg(fr, info_text, (x1, y1 - 8), info_scale, (255, 255, 255),
                             info_thick, bg_color=(0, 0, 0), bg_alpha=0.5)

            # --- 3) Timing overlay (above the info, or top-right of box) ---
            time_text = f"{format_time(seg['start_s'])} - {format_time(seg['end_s'])}"
            put_text_with_bg(fr, time_text, (x1, y1 - 30), info_scale, (0, 255, 255),
                             info_thick, bg_color=(0, 0, 0), bg_alpha=0.5)

            # --- 4) Sub display bar (below box) ---
            sub_text_str = seg_texts.get(si, "")
            if sub_text_str:
                display = f'sub: "{sub_text_str}"'
            else:
                display = f"sub: [segment #{si+1}]"

            # Semi-transparent bar with CJK text
            cjk_size = max(16, int(H / 55))
            bar_h = cjk_size + 16
            bar_y1 = y2 + 2
            bar_y2 = min(H - 1, bar_y1 + bar_h)
            if bar_y2 > bar_y1:
                overlay = fr.copy()
                cv2.rectangle(overlay, (x1, bar_y1), (x2, bar_y2), (40, 40, 40), -1)
                cv2.addWeighted(overlay, 0.7, fr, 0.3, 0, fr)
                # Draw sub text using PIL (supports Chinese)
                put_cjk_text(fr, display, (x1 + 6, bar_y1 + 4), cjk_size,
                             color_rgb=(0, 255, 0))

        # --- Frame counter (top-left) ---
        frame_info = f"Frame: {i}/{n}  Time: {format_time(current_time)}"
        put_text_with_bg(fr, frame_info, (10, 30), info_scale, (200, 200, 200),
                         info_thick, bg_color=(0, 0, 0), bg_alpha=0.4)

        if proc:
            try:
                proc.stdin.write(fr.tobytes())
            except BrokenPipeError:
                raise SystemExit("ffmpeg closed the pipe — check the ffmpeg error above")
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
    print(f"[render-debug] {i} frames -> {out} in {wall:.1f}s ({i / wall:.1f} fps)")


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("-o", "--output")
    ap.add_argument("--work", default="work")
    ap.add_argument("--stages", default="detect,smooth,render",
                    help="Comma-separated stages: detect,smooth,render. Default: all three.")
    ap.add_argument("--no-ocr", action="store_true", help="Skip OCR text recognition")

    # detect params
    g = ap.add_argument_group("detect")
    g.add_argument("--model", default="PP-OCRv5_mobile_det")
    g.add_argument("--device", default=None, help="e.g. gpu:0 or cpu (default: auto)")
    g.add_argument("--engine", default=None)
    g.add_argument("--batch", type=int, default=8)
    g.add_argument("--roi-top", type=float, default=0.55)
    g.add_argument("--det-width", type=int, default=960)
    g.add_argument("--thresh", type=float, default=None)
    g.add_argument("--box-thresh", type=float, default=None)
    g.add_argument("--unclip", type=float, default=None)

    # smooth params (same defaults as sub_boxes.py)
    s = ap.add_argument_group("smooth")
    s.add_argument("--min-score", type=float, default=0.5)
    s.add_argument("--min-h", type=float, default=0.02)
    s.add_argument("--max-h", type=float, default=0.12)
    s.add_argument("--gap-factor", type=float, default=1.5)
    s.add_argument("--min-aspect", type=float, default=1.5)
    s.add_argument("--center-tol", type=float, default=0.25)
    s.add_argument("--band-tol", type=float, default=0.08)
    s.add_argument("--x-tol", type=float, default=0.03)
    s.add_argument("--max-gap", type=int, default=5)
    s.add_argument("--min-len", type=int, default=3)
    s.add_argument("--min-density", type=float, default=0.3)
    s.add_argument("--snap-tol", type=float, default=0.006)
    s.add_argument("--min-width-px", type=int, default=0,
                   help="Min absolute box width in pixels (0 = auto: max(80, 11%% of width))")
    s.add_argument("--pad", type=int, default=4)

    # render params
    r = ap.add_argument_group("render")
    r.add_argument("--color", default="0,255,0")
    r.add_argument("--thickness", type=int, default=0)
    r.add_argument("--nvenc", action="store_true")
    r.add_argument("--crf", type=int, default=18)
    r.add_argument("--preset", default="veryfast")

    a = ap.parse_args()
    P = paths(a)
    stages = [x.strip() for x in a.stages.split(",")]
    meta = probe(a.video)
    timing = {}

    # DETECT
    if "detect" in stages:
        timing["system"] = system_info()
        timing["video"] = meta
        cmd_detect(a, P, timing)
        timing_path = os.path.join(a.work, os.path.splitext(os.path.basename(a.video))[0] + ".timing.json")
        save_json(timing_path, timing)

    # SMOOTH
    segments = None
    if "smooth" in stages:
        segments = cmd_smooth_debug(a, P)
    else:
        sd = load_json(P["segs"])
        if sd is None:
            raise SystemExit(f"Missing {P['segs']} — run smooth first or use --stages smooth,render")
        segments = sd["segments"]

    # OCR (optional)
    seg_texts = {}
    if not a.no_ocr and "render" in stages:
        seg_texts = recognise_segments(a.video, segments, meta)
        # Save OCR results for reuse
        ocr_path = os.path.join(a.work, os.path.splitext(os.path.basename(a.video))[0] + ".ocr_debug.json")
        save_json(ocr_path, seg_texts)
        print(f"[ocr] Results saved to {ocr_path}")
    elif a.no_ocr:
        # Try to load previously saved OCR results
        ocr_path = os.path.join(a.work, os.path.splitext(os.path.basename(a.video))[0] + ".ocr_debug.json")
        loaded = load_json(ocr_path, {})
        if loaded:
            seg_texts = {int(k): v for k, v in loaded.items()}
            print(f"[ocr] Loaded {len(seg_texts)} cached OCR results from {ocr_path}")

    # RENDER
    if "render" in stages:
        cmd_render_debug(a, P, segments, seg_texts)

    # Summary
    print("\n=== DEBUG SUMMARY ===")
    print(f"Segments: {len(segments)}")
    for idx, seg in enumerate(segments):
        x1, y1, x2, y2 = seg["box"]
        bw, bh = x2 - x1, y2 - y1
        text = seg_texts.get(idx, "")
        text_safe = text[:50].encode("ascii", errors="replace").decode("ascii")
        if len(text) > 50:
            text_safe += "..."
        print(f"  #{idx+1:2d}  {format_time(seg['start_s'])} - {format_time(seg['end_s'])}  "
              f"box=[{x1},{y1},{x2},{y2}] {bw}x{bh}  "
              f'sub: "{text_safe}"')

if __name__ == "__main__":
    main()