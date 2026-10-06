#!/usr/bin/env python3
"""
translate.py — Dịch subtitle tiếng Trung (giản thể / phồn thể) sang tiếng Việt.
Kế thừa từ debug_render.py: dùng PaddleOCR nhận diện chữ, sau đó dịch bằng
Google Gemini 2.0 Flash (free tier), render subtitle tiếng Việt lên video và
xuất file .srt.

Cần cài đặt thêm:
    pip install google-genai python-dotenv

Cấu hình qua file .env (cùng thư mục với script) hoặc biến môi trường.
CLI argument luôn được ưu tiên cao hơn .env.

Outputs (cùng thư mục với --output hoặc bên cạnh video gốc):
  1. <output>.mp4      — video đã được render subtitle tiếng Việt
  2. <output>.srt      — file SRT sẵn sàng import vào phần mềm dựng phim

Usage:
  python translate.py "AI Engineer test.mp4" -o "output_vi.mp4"
  python translate.py "AI Engineer test.mp4" -o "output_vi.mp4" --no-ocr --no-render
  python translate.py "AI Engineer test.mp4" -o "output_vi.mp4" --stages smooth,render
  python translate.py "AI Engineer test.mp4" -o "output_vi.mp4" --api-key YOUR_KEY
"""

import argparse
import io
import json
import os
import re
import sys
import time

# Force UTF-8 output — tránh UnicodeEncodeError trên Windows console (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Đọc file .env cùng thư mục với script (trước mọi thứ khác)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
                 override=False)  # override=False: biến OS đã set không bị ghi đè
except ImportError:
    pass  # python-dotenv chưa cài — dùng biến môi trường thông thường

# ── kế thừa toàn bộ pipeline từ debug_render ────────────────────────────────
from debug_render import (
    _find_cjk_font,
    get_cjk_font,
    put_cjk_text,
    ffmpeg_exe,
    probe,
    load_json,
    save_json,
    paths as _base_paths,
    system_info,
    cmd_detect,
    cmd_smooth_debug,
    recognise_segments,
    segments_to_frames,
    format_time,
    put_text_with_bg,
)

import cv2
import numpy as np
import subprocess
from PIL import Image, ImageDraw, ImageFont


# ============================================================================
# Paths helper (thêm các file output riêng cho translate)
# ============================================================================
def paths(a):
    """Mở rộng paths() từ debug_render, thêm các file dành riêng cho dịch thuật."""
    P = _base_paths(a)
    stem = os.path.splitext(os.path.basename(a.video))[0]
    P["ocr"]    = os.path.join(a.work, f"{stem}.ocr_debug.json")
    P["trans"]  = os.path.join(a.work, f"{stem}.translations.json")
    out_base    = os.path.splitext(a.output or (os.path.splitext(a.video)[0] + "_vi.mp4"))[0]
    P["srt"]    = out_base + ".srt"
    P["output"] = out_base + ".mp4"
    return P


# ============================================================================
# SRT helpers
# ============================================================================
def seconds_to_srt_time(s: float) -> str:
    """Chuyển giây → định dạng SRT: HH:MM:SS,mmm"""
    s = max(0.0, s)
    h  = int(s // 3600)
    m  = int((s % 3600) // 60)
    sec = s % 60
    ms  = round((sec - int(sec)) * 1000)
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{int(sec):02d},{ms:03d}"


def export_srt(segments, translations, srt_path):
    """Xuất file SRT từ danh sách segment và bảng dịch."""
    lines = []
    for idx, seg in enumerate(segments):
        vi_text = translations.get(idx, translations.get(str(idx), ""))
        if not vi_text:
            continue
        t_start = seconds_to_srt_time(seg["start_s"])
        t_end   = seconds_to_srt_time(seg["end_s"])
        lines.append(str(idx + 1))
        lines.append(f"{t_start} --> {t_end}")
        lines.append(vi_text)
        lines.append("")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[srt] Đã xuất {sum(1 for l in lines if '-->' in l)} dòng → {srt_path}")


# ============================================================================
# Gemini translation
# ============================================================================
_SYSTEM_PROMPT = """Bạn là dịch giả phụ đề phim chuyên nghiệp.
Dịch tiếng Trung (giản thể hoặc phồn thể) sang tiếng Việt.

Quy tắc:
- Dịch tự nhiên, như người Việt nói thật trong phim.
- Giữ tên riêng nếu không có bản dịch chuẩn.
- Phụ đề ngắn gọn (≤ 42 ký tự/dòng).
- KHÔNG thêm giải thích. KHÔNG markdown. KHÔNG nằm ngoài mảng JSON.

Trả về ĐÚNG mảng JSON, không có gì thêm:
[{"id":<int>,"vi":"<dịch>"},{"id":<int>,"vi":"<dịch>"},...]"""

_USER_TEMPLATE = """Dịch các phụ đề sau từ tiếng Trung sang tiếng Việt:

{items}"""


def _build_batch_prompt(batch):
    """batch: list of (idx, zh_text)"""
    items_str = "\n".join(f'{i+1}. [ID:{idx}] {zh}' for i, (idx, zh) in enumerate(batch))
    return _USER_TEMPLATE.format(items=items_str)


def _parse_translation_response(text: str) -> dict:
    """
    Parse JSON từ Gemini response. Rất robust:
    - Thử parse toàn bộ mảng JSON trước
    - Nếu bị cắt đứt (Unterminated...), dùng regex cứu từng object đã hoàn chỉnh
    """
    # --- Bước 1: cố định parse toàn bộ ---
    # Tìm mảng JSON trong response (bỏ phần text thừa của model)
    arr_match = re.search(r'\[.*\]', text, re.DOTALL)
    if arr_match:
        candidate = arr_match.group(0)
        try:
            data = json.loads(candidate)
            result = {int(item["id"]): item["vi"]
                      for item in data if "id" in item and "vi" in item}
            if result:
                return result
        except json.JSONDecodeError:
            pass  # bị cắt, thử cách 2

    # --- Bước 2: cứu từng object đã hoàn chỉnh bằng regex ---
    # Match các object {"id": N, "vi": "..."}  (chỉ lấy các object kông bị cắt giữa)
    pattern = re.compile(
        r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"vi"\s*:\s*("(?:[^"\\]|\\.)*")\s*\}',
        re.DOTALL
    )
    result = {}
    for m in pattern.finditer(text):
        try:
            idx = int(m.group(1))
            vi  = json.loads(m.group(2))   # decode escape sequences
            result[idx] = vi
        except Exception:
            continue

    if not result:
        # Log 500 ký tự đầu để debug
        preview = text[:500].replace('\n', ' ')
        print(f"[translate] Không parse được JSON. Raw (500 chars): {preview}")
    else:
        print(f"[translate] Parse bất buốc được {len(result)} item (JSON bị cắt).")
    return result


def _list_available_models(client) -> list:
    """Lấy danh sách model Gemini hỗ trợ generateContent."""
    try:
        models = []
        for m in client.models.list():
            name = getattr(m, 'name', '') or ''
            # Chỉ lấy tên ngắn (bỏ prefix 'models/')
            short = name.removeprefix('models/')
            supported = getattr(m, 'supported_actions', []) or []
            if 'generateContent' in supported and short:
                models.append(short)
        return sorted(models)
    except Exception:
        return []


def _is_model_error(exc) -> bool:
    """Kiểm tra exception có phải lỗi tên model sai không."""
    msg = str(exc).lower()
    return ('invalid_argument' in msg and 'model' in msg) or \
           'unexpected model name' in msg or \
           'model not found' in msg or \
           'not supported' in msg


def translate_batch_gemini(batch, client, model_name: str, max_retries: int = 3) -> dict:
    """
    Dịch một batch segments bằng Gemini.
    batch: list of (idx, zh_text)
    Trả về dict {idx -> vi_text}
    """
    prompt = _build_batch_prompt(batch)
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "system_instruction": _SYSTEM_PROMPT,
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "max_output_tokens": 4096,  # tăng để tránh bị cắt giữa JSON
                }
            )
            raw = response.text.strip()
            result = _parse_translation_response(raw)
            if result:
                return result
            print(f"[translate] Batch rỗng, thử lại ({attempt+1}/{max_retries})...")
        except Exception as e:
            # Lỗi tên model sai → không cần retry, báo ngay và thoát
            if _is_model_error(e):
                raise
            wait = 2 ** attempt
            print(f"[translate] Lỗi API: {e}  — chờ {wait}s rồi thử lại...")
            time.sleep(wait)
    return {}


def translate_all(seg_texts: dict, api_key: str, model_name: str,
                  batch_size: int = 20, delay: float = 1.0) -> dict:
    """
    Dịch toàn bộ seg_texts {idx -> zh_text} sang tiếng Việt.
    Trả về dict {idx -> vi_text}.
    """
    try:
        from google import genai
    except ImportError:
        raise SystemExit(
            "[translate] Chưa cài google-genai.\n"
            "  Chạy: pip install google-genai"
        )

    client = genai.Client(api_key=api_key)

    # ── Validate model name trước khi gửi batch thật ─────────────────────────
    print(f"[translate] Kiểm tra model '{model_name}'...")
    try:
        client.models.generate_content(
            model=model_name,
            contents="test",
            config={"max_output_tokens": 1}
        )
    except Exception as probe_err:
        if _is_model_error(probe_err):
            avail = _list_available_models(client)
            lines = []
            lines.append(f"\n[translate] ❌ Model '{model_name}' không hợp lệ hoặc không tồn tại.")
            if avail:
                lines.append("[translate] Các model bạn có thể dùng với API key này:")
                for m in avail:
                    marker = " ← (khuyến nghị)" if "flash" in m and "lite" in m else ""
                    lines.append(f"  • {m}{marker}")
                lines.append("\n[translate] Sửa GEMINI_MODEL trong .env hoặc dùng --model <tên>.")
            else:
                lines.append("[translate] Không lấy được danh sách model. Kiểm tra lại API key và kết nối.")
            raise SystemExit("\n".join(lines))
        # Lỗi khác (rate limit, network...) → bỏ qua, tiến hành dịch thật
        print(f"[translate] Cảnh báo probe: {probe_err} — tiếp tục...")

    # Chỉ dịch những segment có text (bỏ qua segment rỗng)
    valid = [(idx, text) for idx, text in sorted(seg_texts.items(), key=lambda x: int(x[0]))
             if text and text.strip()]

    if not valid:
        print("[translate] Không có text nào để dịch.")
        return {}

    print(f"[translate] Dịch {len(valid)} segment với model={model_name}, batch={batch_size}...")
    translations = {}
    batches = [valid[i:i+batch_size] for i in range(0, len(valid), batch_size)]

    for b_idx, batch in enumerate(batches):
        print(f"[translate] Batch {b_idx+1}/{len(batches)} ({len(batch)} items)...")
        try:
            result = translate_batch_gemini(batch, client, model_name)
        except Exception as e:
            if _is_model_error(e):
                # Không nên xảy ra sau probe, nhưng xử lý phòng thủ
                raise SystemExit(f"[translate] Lỗi model '{model_name}': {e}")
            raise
        translations.update(result)

        missing = [idx for idx, _ in batch if idx not in translations]
        if missing:
            print(f"[translate]  ⚠ Thiếu bản dịch cho idx: {missing}")

        if b_idx < len(batches) - 1:
            time.sleep(delay)  # tránh rate limit free tier

    ok  = sum(1 for _, vi in translations.items() if vi)
    print(f"[translate] Hoàn thành: {ok}/{len(valid)} segment được dịch.")
    return translations


# ============================================================================
# Render: ghi subtitle tiếng Việt lên video
# ============================================================================
def cmd_render_translated(a, P, segments, translations):
    """
    Render video với subtitle tiếng Việt (thay vì debug overlay).
    - Subtitle bar màu đen trong suốt ở dưới mỗi bounding box
    - Chữ Việt màu trắng, rõ ràng
    - Không hiển thị các overlay debug (coordinates, timing)
    """
    meta = probe(a.video)
    W, H, fps, n = meta["width"], meta["height"], meta["fps"], meta["n_frames"]
    per = segments_to_frames(segments, max(n, len(segments) and segments[-1]["end"] + 2) + 1)
    out = P["output"]

    ff = ffmpeg_exe()
    proc, vw = None, None
    if ff:
        v = (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19"] if a.nvenc
             else ["-c:v", "libx264", "-preset", a.preset, "-crf", str(a.crf)])
        cmd = [ff, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{W}x{H}", "-r", f"{fps:.6f}",
               "-i", "-",
               "-i", a.video,
               "-map", "0:v:0", "-map", "1:a?",
               *v, "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k",
               "-shortest", out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    else:
        print("[render] ffmpeg không tìm thấy — fallback OpenCV mp4v (không có audio)")
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # Font size dựa theo chiều cao video
    cjk_size   = max(22, int(H / 40))
    bar_pad    = 8
    bar_h      = cjk_size + bar_pad * 2

    cap = cv2.VideoCapture(a.video)
    t0  = time.perf_counter()
    i   = 0

    while True:
        ok, fr = cap.read()
        if not ok:
            break

        items = per[i] if i < len(per) else []

        for box, si in items:
            x1, y1, x2, y2 = box
            vi_text = translations.get(si, translations.get(str(si), ""))
            if not vi_text:
                continue

            # --- Vẽ khung viền phụ đề (mỏng, màu xanh lá) ---
            cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 220, 80), 2)

            # --- Subtitle bar bên dưới box ---
            bar_y1 = y2 + 4
            bar_y2 = min(H - 1, bar_y1 + bar_h)

            if bar_y2 > bar_y1:
                overlay = fr.copy()
                cv2.rectangle(overlay, (x1, bar_y1), (x2, bar_y2), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.75, fr, 0.25, 0, fr)
                put_cjk_text(fr, vi_text,
                             (x1 + bar_pad, bar_y1 + bar_pad),
                             cjk_size,
                             color_rgb=(255, 255, 255))

        if proc:
            try:
                proc.stdin.write(fr.tobytes())
            except BrokenPipeError:
                raise SystemExit("ffmpeg đóng pipe — kiểm tra lỗi ffmpeg")
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
    print(f"[render-vi] {i} frames → {out}  ({wall:.1f}s, {i / max(wall,0.001):.1f} fps)")


# ============================================================================
# Main
# ============================================================================
def _env(key, default=None, cast=None):
    """Đọc biến môi trường (đã được load từ .env) với kiểu tùy chọn."""
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    if cast is bool:
        return val.strip().lower() in ("1", "true", "yes", "on")
    if cast is not None:
        try:
            return cast(val)
        except (ValueError, TypeError):
            return default
    return val


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("video", help="Video đầu vào (tiếng Trung)")
    ap.add_argument("-o", "--output",
                    help="File video đầu ra (mặc định: <tên_gốc>_vi.mp4)")
    ap.add_argument("--work",    default="work",
                    help="Thư mục lưu cache (default: work/)")
    ap.add_argument("--stages",  default="detect,smooth,ocr,translate,render",
                    help="Các bước xử lý, phân cách bởi dấu phẩy.")
    ap.add_argument("--no-ocr",  action="store_true",
                    help="Bỏ qua OCR, dùng cache nếu có")
    ap.add_argument("--no-render", action="store_true",
                    help="Chỉ dịch + xuất SRT, không render video")
    ap.add_argument("--api-key",
                    default=None,   # resolve sau từ .env / os.environ
                    help="Gemini API key (ưu tiên hơn GEMINI_API_KEY trong .env)")
    ap.add_argument("--model",
                    default=_env("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                    # Free-tier models (2026): gemini-3.5-flash-lite, gemini-3.8-flash
                    help="Tên model Gemini [env: GEMINI_MODEL]")
    ap.add_argument("--batch-size",
                    type=int, default=_env("TRANSLATE_BATCH_SIZE", 5, int),
                    help="So segment moi lan gui API [env: TRANSLATE_BATCH_SIZE] (default: 5)")
    ap.add_argument("--delay",
                    type=float, default=_env("TRANSLATE_DELAY", 13.0, float),
                    help="Cho giua cac batch (giay) [env: TRANSLATE_DELAY] (default: 13s cho free tier 5rpm)")

    # detect params (kế thừa từ debug_render)
    g = ap.add_argument_group("detect")
    g.add_argument("--det-model",   default="PP-OCRv5_mobile_det",
                   dest="model_det",
                   help="PaddleOCR detection model")
    g.add_argument("--device",
                   default=_env("OCR_DEVICE") or None,
                   help="e.g. gpu:0 / cpu (auto nếu để trống) [env: OCR_DEVICE]")
    g.add_argument("--engine",      default=None)
    g.add_argument("--batch",       type=int,   default=8)
    g.add_argument("--roi-top",
                   type=float, default=_env("OCR_ROI_TOP", 0.55, float),
                   help="% chiều cao bắt đầu ROI [env: OCR_ROI_TOP]")
    g.add_argument("--det-width",
                   type=int,   default=_env("OCR_DET_WIDTH", 960, int),
                   help="Độ rộng ảnh detect [env: OCR_DET_WIDTH]")
    g.add_argument("--thresh",      type=float, default=None)
    g.add_argument("--box-thresh",  type=float, default=None)
    g.add_argument("--unclip",      type=float, default=None)

    # smooth params
    s = ap.add_argument_group("smooth")
    s.add_argument("--min-score",   type=float, default=0.5)
    s.add_argument("--min-h",       type=float, default=0.02)
    s.add_argument("--max-h",       type=float, default=0.12)
    s.add_argument("--gap-factor",  type=float, default=1.5)
    s.add_argument("--min-aspect",  type=float, default=1.5)
    s.add_argument("--center-tol",  type=float, default=0.25)
    s.add_argument("--band-tol",    type=float, default=0.08)
    s.add_argument("--x-tol",       type=float, default=0.03)
    s.add_argument("--max-gap",     type=int,   default=5)
    s.add_argument("--min-len",     type=int,   default=3)
    s.add_argument("--min-density", type=float, default=0.3)
    s.add_argument("--snap-tol",    type=float, default=0.006)
    s.add_argument("--min-width-px",type=int,   default=0)
    s.add_argument("--pad",         type=int,   default=4)

    # render params
    r = ap.add_argument_group("render")
    r.add_argument("--nvenc",  action="store_true",
                   default=_env("RENDER_NVENC", False, bool))
    r.add_argument("--crf",    type=int,  default=_env("RENDER_CRF", 18, int),
                   help="CRF libx264 [env: RENDER_CRF]")
    r.add_argument("--preset", default=_env("RENDER_PRESET", "veryfast"),
                   help="Preset libx264 [env: RENDER_PRESET]")

    a = ap.parse_args()

    # ── Tách biệt: lưu Gemini model TRƯỚC khi a.model bị ghi đè ─────────────
    gemini_model = a.model   # tên model Gemini để dịch
    a.model = a.model_det    # cmd_detect() từ debug_render cần a.model = PaddleOCR model

    # ── Resolve API key: CLI > .env / os.environ ──────────────────────────────
    api_key = a.api_key or _env("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        raise SystemExit(
            "[translate] Chưa cấu hình GEMINI_API_KEY.\n"
            "  Cách 1: Sửa GEMINI_API_KEY trong file .env\n"
            "  Cách 2: --api-key YOUR_KEY\n"
            "  Cách 3: $env:GEMINI_API_KEY='YOUR_KEY'  (PowerShell)"
        )

    # In cấu hình hiệu lực
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    print(f"[config] .env: {env_file} ({'found' if os.path.exists(env_file) else 'not found'})")
    print(f"[config] gemini_model={gemini_model}  batch={a.batch_size}  delay={a.delay}s")
    print(f"[config] ocr_model={a.model}  roi_top={a.roi_top}  det_width={a.det_width}  device={a.device or 'auto'}")
    print(f"[config] crf={a.crf}  preset={a.preset}  nvenc={a.nvenc}")

    P      = paths(a)
    stages = [x.strip() for x in a.stages.split(",")]
    meta   = probe(a.video)
    timing = {}

    # ── DETECT ──────────────────────────────────────────────────────────────
    if "detect" in stages:
        timing["system"] = system_info()
        timing["video"]  = meta
        cmd_detect(a, P, timing)
        timing_path = os.path.join(
            a.work,
            os.path.splitext(os.path.basename(a.video))[0] + ".timing.json"
        )
        save_json(timing_path, timing)

    # ── SMOOTH ───────────────────────────────────────────────────────────────
    segments = None
    if "smooth" in stages:
        segments = cmd_smooth_debug(a, P)
    else:
        sd = load_json(P["segs"])
        if sd is None:
            raise SystemExit(
                f"Thiếu {P['segs']} — chạy smooth trước "
                "hoặc dùng --stages smooth,ocr,translate,render"
            )
        segments = sd["segments"]

    print(f"[main] {len(segments)} subtitle segments.")

    # ── OCR ──────────────────────────────────────────────────────────────────
    seg_texts = {}
    if "ocr" in stages and not a.no_ocr:
        seg_texts = recognise_segments(a.video, segments, meta)
        save_json(P["ocr"], seg_texts)
        print(f"[ocr] Lưu cache → {P['ocr']}")
    else:
        loaded = load_json(P["ocr"], {})
        if loaded:
            seg_texts = {int(k): v for k, v in loaded.items()}
            print(f"[ocr] Dùng cache OCR ({len(seg_texts)} items) từ {P['ocr']}")
        else:
            print("[ocr] Không có cache OCR, bỏ qua.")

    # ── TRANSLATE ─────────────────────────────────────────────────────────────
    translations = {}
    if "translate" in stages:
        # Thử load cache dịch trước
        cached_trans = load_json(P["trans"], {})
        if cached_trans:
            translations = {int(k): v for k, v in cached_trans.items()}
            print(f"[translate] Đã load cache dịch ({len(translations)} items) từ {P['trans']}")

        # Tìm những segment chưa được dịch
        untranslated = {idx: text for idx, text in seg_texts.items()
                        if text and idx not in translations}
        if untranslated:
            new_trans = translate_all(
                untranslated, api_key, gemini_model,
                batch_size=a.batch_size, delay=a.delay
            )
            translations.update(new_trans)
            # Lưu cache
            save_json(P["trans"], translations)
            print(f"[translate] Lưu cache → {P['trans']}")
        else:
            print("[translate] Tất cả segment đã có bản dịch (dùng cache).")
    else:
        loaded = load_json(P["trans"], {})
        if loaded:
            translations = {int(k): v for k, v in loaded.items()}
            print(f"[translate] Dùng cache dịch ({len(translations)} items).")

    # ── EXPORT SRT ───────────────────────────────────────────────────────────
    export_srt(segments, translations, P["srt"])

    # ── RENDER ───────────────────────────────────────────────────────────────
    if "render" in stages and not a.no_render:
        cmd_render_translated(a, P, segments, translations)

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("\n=== TRANSLATION SUMMARY ===")
    print(f"Segments   : {len(segments)}")
    print(f"OCR ok     : {sum(1 for v in seg_texts.values() if v)}")
    print(f"Translated : {sum(1 for v in translations.values() if v)}")
    print(f"SRT file   : {P['srt']}")
    if not a.no_render:
        print(f"Video out  : {P['output']}")
    print()
    for idx, seg in enumerate(segments):
        zh   = seg_texts.get(idx, "")
        vi   = translations.get(idx, "")
        zh_s = zh[:40].encode("ascii", errors="replace").decode("ascii") if zh else "(no ocr)"
        vi_s = vi[:40] if vi else "(not translated)"
        if len(zh) > 40:
            zh_s += "..."
        if len(vi) > 40:
            vi_s += "..."
        print(f"  #{idx+1:2d}  {format_time(seg['start_s'])} - {format_time(seg['end_s'])}")
        print(f"       ZH: {zh_s}")
        print(f"       VI: {vi_s}")


if __name__ == "__main__":
    main()
