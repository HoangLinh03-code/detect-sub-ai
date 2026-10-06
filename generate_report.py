#!/usr/bin/env python3
"""
Generate performance report Excel from timing data.
Usage: python generate_report.py
"""
import json
import os

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
except ImportError:
    print("Installing openpyxl...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter


def load_timing(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def style_header(ws, row, max_col):
    """Apply header styling."""
    header_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
    for col in range(1, max_col + 1):
        cell = ws.cell(row=row, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def style_title(ws, row, col, text, merge_end_col=None):
    """Apply title styling."""
    cell = ws.cell(row=row, column=col, value=text)
    cell.font = Font(name="Arial", size=14, bold=True, color="1F3864")
    cell.alignment = Alignment(horizontal="left", vertical="center")
    if merge_end_col:
        ws.merge_cells(start_row=row, start_column=col, end_row=row, end_column=merge_end_col)


def apply_borders(ws, min_row, max_row, min_col, max_col):
    """Apply thin borders to a range."""
    thin = Side(style="thin", color="B4C6E7")
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            ws.cell(row=row, column=col).border = Border(
                left=thin, right=thin, top=thin, bottom=thin
            )


def alt_row_fill(ws, min_row, max_row, min_col, max_col):
    """Alternate row shading."""
    light = PatternFill(start_color="D6E4F0", end_color="D6E4F0", fill_type="solid")
    for row in range(min_row, max_row + 1):
        if (row - min_row) % 2 == 1:
            for col in range(min_col, max_col + 1):
                ws.cell(row=row, column=col).fill = light


def auto_width(ws, min_col=1, max_col=None, padding=3):
    """Auto-fit column widths."""
    if max_col is None:
        max_col = ws.max_column
    for col in range(min_col, max_col + 1):
        max_len = 0
        for row in ws.iter_rows(min_col=col, max_col=col, values_only=False):
            for cell in row:
                if cell.value:
                    # Handle multi-line
                    for line in str(cell.value).split("\n"):
                        max_len = max(max_len, len(line))
        ws.column_dimensions[get_column_letter(col)].width = min(max_len + padding, 50)


def create_report(timing_path, output_path):
    data = load_timing(timing_path)
    sys_info = data.get("system", {})
    video_info = data.get("video", {})
    detect_info = data.get("detect", {})

    # Derived values
    fps_video = video_info.get("fps", 30.0)
    n_frames = video_info.get("n_frames", 0)
    duration_s = n_frames / fps_video if fps_video else 0
    detect_wall = detect_info.get("wall_s", 0)
    detect_model = detect_info.get("model_s", 0)
    detect_fps = n_frames / detect_wall if detect_wall else 0
    smooth_wall = 0.025  # from last run
    render_wall = 63.3   # from last run
    total_wall = detect_wall + smooth_wall + render_wall
    total_fps = n_frames / total_wall if total_wall else 0
    rtf = total_wall / duration_s if duration_s else 0
    latency_per_frame_ms = (detect_wall / n_frames * 1000) if n_frames else 0

    wb = openpyxl.Workbook()

    # =========================================================================
    # SHEET 1: Cấu hình hệ thống & Thông số video
    # =========================================================================
    ws1 = wb.active
    ws1.title = "Cấu Hình Hệ Thống"
    ws1.sheet_properties.tabColor = "2F5496"

    style_title(ws1, 1, 1, "BÁO CÁO HIỆU NĂNG — SUBTITLE BOUNDING BOX DETECTION", 4)
    ws1.row_dimensions[1].height = 30

    style_title(ws1, 3, 1, "1. Cấu hình máy tính sử dụng", 4)

    headers = ["Thành phần", "Chi tiết"]
    for col, h in enumerate(headers, 1):
        ws1.cell(row=5, column=col, value=h)
    style_header(ws1, 5, 2)

    gpu_raw = sys_info.get("gpu", "N/A")
    gpu_parts = gpu_raw.split(",") if gpu_raw else ["N/A"]
    gpu_name = gpu_parts[0].strip()
    gpu_vram = gpu_parts[1].strip() if len(gpu_parts) > 1 else ""

    config_rows = [
        ("CPU", sys_info.get("cpu", "N/A")),
        ("CPU Cores / Threads", f'{sys_info.get("cpu_count", "N/A")} logical cores'),
        ("GPU", f"{gpu_name} ({gpu_vram})" if gpu_vram else gpu_name),
        ("GPU Driver", gpu_parts[2].strip() if len(gpu_parts) > 2 else "N/A"),
        ("RAM", f'{sys_info.get("ram_gb", "N/A")} GB'),
        ("Hệ điều hành", sys_info.get("os", "N/A")),
        ("Python", sys_info.get("python", "N/A")),
        ("OpenCV", sys_info.get("opencv", "N/A")),
        ("PaddlePaddle", sys_info.get("paddle", "N/A")),
        ("PaddleOCR", sys_info.get("paddleocr", "N/A")),
    ]
    for i, (k, v) in enumerate(config_rows):
        ws1.cell(row=6 + i, column=1, value=k).font = Font(bold=True, name="Arial", size=10)
        ws1.cell(row=6 + i, column=2, value=v).font = Font(name="Arial", size=10)

    apply_borders(ws1, 5, 5 + len(config_rows), 1, 2)
    alt_row_fill(ws1, 6, 5 + len(config_rows), 1, 2)

    # Video info
    r = 6 + len(config_rows) + 2
    style_title(ws1, r, 1, "2. Thông số video đầu vào", 4)
    r += 2
    headers2 = ["Thông số", "Giá trị"]
    for col, h in enumerate(headers2, 1):
        ws1.cell(row=r, column=col, value=h)
    style_header(ws1, r, 2)

    video_rows = [
        ("Tên file", "AI Engineer test.mp4"),
        ("Độ phân giải", f'{video_info.get("width", 0)}x{video_info.get("height", 0)}'),
        ("FPS gốc", f'{fps_video:.1f} fps'),
        ("Tổng số frame", f'{n_frames:,}'),
        ("Thời lượng", f'{duration_s:.1f} giây ({duration_s/60:.1f} phút)'),
        ("Model phát hiện", detect_info.get("model", "N/A")),
        ("Detection width", f'{detect_info.get("det_width", 0)} px'),
        ("Batch size", detect_info.get("batch", 0)),
        ("Device", detect_info.get("device", "auto")),
    ]
    for i, (k, v) in enumerate(video_rows):
        ws1.cell(row=r + 1 + i, column=1, value=k).font = Font(bold=True, name="Arial", size=10)
        ws1.cell(row=r + 1 + i, column=2, value=str(v)).font = Font(name="Arial", size=10)

    apply_borders(ws1, r, r + len(video_rows), 1, 2)
    alt_row_fill(ws1, r + 1, r + len(video_rows), 1, 2)

    auto_width(ws1)

    # =========================================================================
    # SHEET 2: Thời gian xử lý thực tế
    # =========================================================================
    ws2 = wb.create_sheet("Hiệu Năng Thực Tế")
    ws2.sheet_properties.tabColor = "548235"

    style_title(ws2, 1, 1, "3. Thời gian xử lý thực tế", 6)
    ws2.row_dimensions[1].height = 30

    # Stage breakdown
    headers3 = ["Giai đoạn (Stage)", "Thời gian (s)", "FPS", "Ghi chú"]
    r = 3
    for col, h in enumerate(headers3, 1):
        ws2.cell(row=r, column=col, value=h)
    style_header(ws2, r, len(headers3))

    stages = [
        ("1. Detect (Text Detection)", detect_wall, detect_fps,
         f"Model: {detect_info.get('model','N/A')}, GPU inference"),
        ("   ├─ Model inference", detect_model, n_frames / detect_model if detect_model else 0,
         "GPU (CUDA) — phần nặng nhất"),
        ("   ├─ Decode + ROI crop", detect_info.get("decode_roi_thread_s", 0), "",
         "CPU thread riêng, song song với inference"),
        ("   └─ Warm-up (excluded)", detect_info.get("warmup_s", 0), "",
         "CUDA init + cudnn autotune, chạy 1 lần"),
        ("2. Smooth (Post-processing)", smooth_wall, n_frames / smooth_wall if smooth_wall else 0,
         "CPU only — filter, track, merge segments"),
        ("3. Render (Draw + Encode)", render_wall, n_frames / render_wall if render_wall else 0,
         "CPU draw + ffmpeg libx264 encode"),
        ("", "", "", ""),
        ("TỔNG CỘNG (pipeline)", round(total_wall, 2), round(total_fps, 1),
         f"RTF = {rtf:.2f} (xử lý / thời lượng video)"),
    ]

    for i, (stage, time_s, fps_val, note) in enumerate(stages):
        row = r + 1 + i
        ws2.cell(row=row, column=1, value=stage).font = Font(name="Arial", size=10,
                                                              bold=(i == len(stages) - 1))
        if time_s != "":
            c = ws2.cell(row=row, column=2, value=round(float(time_s), 3) if time_s else "")
            c.font = Font(name="Arial", size=10, bold=(i == len(stages) - 1))
            c.number_format = "0.000"
        if fps_val != "":
            c = ws2.cell(row=row, column=3, value=round(float(fps_val), 1) if fps_val else "")
            c.font = Font(name="Arial", size=10, bold=(i == len(stages) - 1))
            c.number_format = "0.0"
        ws2.cell(row=row, column=4, value=note).font = Font(name="Arial", size=10,
                                                             color="666666" if i < len(stages) - 1 else "000000")

    apply_borders(ws2, r, r + len(stages), 1, len(headers3))
    alt_row_fill(ws2, r + 1, r + len(stages), 1, len(headers3))

    # Highlight total row
    total_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    for col in range(1, len(headers3) + 1):
        ws2.cell(row=r + len(stages), column=col).fill = total_fill

    # Key metrics summary
    r2 = r + len(stages) + 3
    style_title(ws2, r2, 1, "Tóm tắt chỉ số chính", 6)
    r2 += 2
    metrics_headers = ["Chỉ số", "Giá trị", "Đơn vị"]
    for col, h in enumerate(metrics_headers, 1):
        ws2.cell(row=r2, column=col, value=h)
    style_header(ws2, r2, len(metrics_headers))

    metrics = [
        ("Processing FPS (tổng pipeline)", round(total_fps, 1), "fps"),
        ("Processing FPS (detect only)", round(detect_fps, 1), "fps"),
        ("Latency per frame (detect)", round(latency_per_frame_ms, 2), "ms/frame"),
        ("Tổng thời gian xử lý", round(total_wall, 2), "giây"),
        ("Thời lượng video", round(duration_s, 1), "giây"),
        ("Real-Time Factor (RTF)", round(rtf, 2), "× thời gian thực"),
        ("Segments phát hiện", 46, "subtitle segments"),
    ]
    for i, (k, v, u) in enumerate(metrics):
        row = r2 + 1 + i
        ws2.cell(row=row, column=1, value=k).font = Font(bold=True, name="Arial", size=10)
        c = ws2.cell(row=row, column=2, value=v)
        c.font = Font(name="Arial", size=11, bold=True, color="2F5496")
        c.alignment = Alignment(horizontal="center")
        ws2.cell(row=row, column=3, value=u).font = Font(name="Arial", size=10, color="666666")

    apply_borders(ws2, r2, r2 + len(metrics), 1, len(metrics_headers))
    alt_row_fill(ws2, r2 + 1, r2 + len(metrics), 1, len(metrics_headers))

    auto_width(ws2)

    # =========================================================================
    # SHEET 3: Ước lượng hiệu năng trên 4 nhóm thiết bị
    # =========================================================================
    ws3 = wb.create_sheet("Ước Lượng Thiết Bị")
    ws3.sheet_properties.tabColor = "BF8F00"

    style_title(ws3, 1, 1, "4. Bảng ước lượng thời gian xử lý trên các nhóm thiết bị", 7)
    ws3.row_dimensions[1].height = 30

    # Reference: actual device = RTX 3050 Laptop 4GB, i7-12700H
    # Detect FPS actual = detect_fps on RTX 3050 4GB
    #
    # Estimation methodology:
    #   - Detect is GPU-bound (model inference). Scale by relative GPU performance.
    #   - Smooth is CPU-bound, negligible (<0.03s). Scale by CPU slightly.
    #   - Render is CPU+encode bound. Scale by CPU/encode capability.
    #
    # Relative GPU perf (approx FP32 TFLOPS for text detection workloads):
    #   RTX 3050 Laptop 4GB ≈ 7.6 TFLOPS   (baseline = 1.0x)
    #   CPU only (i5/i7 12th) ≈ ~0.8 TFLOPS equivalent for PaddleOCR → ~0.1x GPU
    #   RTX 3050 Desktop 8GB ≈ 9.1 TFLOPS   → ~1.2x
    #   RTX 3060 12GB ≈ 12.7 TFLOPS         → ~1.7x
    #   RTX 4070 ≈ 29.1 TFLOPS              → ~3.8x
    #   RTX 4090 ≈ 82.6 TFLOPS              → ~10.8x (but memory bandwidth limited → ~6x practical)
    #   Tesla T4 ≈ 8.1 TFLOPS               → ~1.1x
    #   Apple M1 (ANE/GPU) ≈ ~2.6 TFLOPS    → ~0.34x (CoreML, not CUDA)
    #   Apple M2 ≈ ~3.6 TFLOPS              → ~0.47x
    #   Apple M3 Pro ≈ ~5.0 TFLOPS          → ~0.66x
    #   Apple M3 Max ≈ ~14 TFLOPS           → ~1.8x

    r = 3
    # Methodology note
    note_cell = ws3.cell(row=r, column=1,
                         value="Phương pháp ước lượng: Scale GPU TFLOPS tương đối so với thiết bị benchmark "
                               f"(RTX 3050 Laptop 4GB = {detect_fps:.1f} fps detect). "
                               "Detect là GPU-bound. Smooth là CPU (< 0.1s). "
                               "Render scale theo CPU/encoder capability.")
    note_cell.font = Font(name="Arial", size=9, italic=True, color="666666")
    ws3.merge_cells(start_row=r, start_column=1, end_row=r, end_column=7)
    note_cell.alignment = Alignment(wrap_text=True)
    ws3.row_dimensions[r].height = 36

    r = 5
    headers4 = [
        "Nhóm Thiết Bị",
        "Cấu Hình Tham Chiếu",
        "Detect\nFPS",
        "Detect\nLatency (ms/frame)",
        "Pipeline\nFPS",
        "Tổng thời gian\nxử lý (s)",
        "RTF\n(xử lý/video)",
    ]
    for col, h in enumerate(headers4, 1):
        ws3.cell(row=r, column=col, value=h)
    style_header(ws3, r, len(headers4))
    ws3.row_dimensions[r].height = 36

    # Device groups with estimated multipliers vs our RTX 3050 Laptop
    # detect_fps is our baseline
    devices = [
        {
            "group": "★ Thiết bị benchmark (thực tế đo)",
            "config": f"Intel i7-12700H, RTX 3050 Laptop 4GB\nRAM 24GB, Windows 10",
            "det_mult": 1.0,
            "render_mult": 1.0,
            "is_actual": True,
        },
        {
            "group": "Máy văn phòng (CPU only)",
            "config": "Intel Core i5/i7 thế hệ 12-14\nRAM 16GB, Không có GPU rời",
            "det_mult": 0.10,  # CPU-only PaddleOCR ~10x slower
            "render_mult": 0.85,
            "is_actual": False,
        },
        {
            "group": "Laptop/PC Đồ họa",
            "config": "NVIDIA RTX 3050 8GB / RTX 3060 12GB\nCPU i7, RAM 16-32GB",
            "det_mult": 1.45,  # average of 3050 desktop (1.2) and 3060 (1.7)
            "render_mult": 1.1,
            "is_actual": False,
        },
        {
            "group": "Máy trạm / Server",
            "config": "NVIDIA RTX 4070 / RTX 4090\nhoặc Tesla T4, RAM 32-64GB",
            "det_mult": 3.5,  # conservative for 4070~4090 range
            "render_mult": 1.5,
            "is_actual": False,
        },
        {
            "group": "Apple Silicon",
            "config": "Apple M1 / M2 / M3 Pro hoặc Max\nRAM 16-64GB (unified memory)",
            "det_mult": 0.55,  # average M1~M3 range, MPS backend
            "render_mult": 1.0,
            "is_actual": False,
        },
    ]

    actual_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")

    for i, dev in enumerate(devices):
        row = r + 1 + i
        est_det_fps = detect_fps * dev["det_mult"]
        est_det_latency = 1000.0 / est_det_fps if est_det_fps else 0
        est_det_wall = n_frames / est_det_fps if est_det_fps else 0
        est_render_wall = render_wall / dev["render_mult"]
        est_total = est_det_wall + smooth_wall + est_render_wall
        est_pipeline_fps = n_frames / est_total if est_total else 0
        est_rtf = est_total / duration_s if duration_s else 0

        ws3.cell(row=row, column=1, value=dev["group"]).font = Font(
            name="Arial", size=10, bold=True,
            color="548235" if dev["is_actual"] else "000000")
        ws3.cell(row=row, column=2, value=dev["config"]).font = Font(name="Arial", size=9)
        ws3.cell(row=row, column=2).alignment = Alignment(wrap_text=True, vertical="center")
        ws3.row_dimensions[row].height = 36

        c3 = ws3.cell(row=row, column=3, value=round(est_det_fps, 1))
        c3.font = Font(name="Arial", size=11, bold=True)
        c3.alignment = Alignment(horizontal="center")
        c3.number_format = "0.0"

        c4 = ws3.cell(row=row, column=4, value=round(est_det_latency, 1))
        c4.font = Font(name="Arial", size=11)
        c4.alignment = Alignment(horizontal="center")
        c4.number_format = "0.0"

        c5 = ws3.cell(row=row, column=5, value=round(est_pipeline_fps, 1))
        c5.font = Font(name="Arial", size=11, bold=True, color="2F5496")
        c5.alignment = Alignment(horizontal="center")
        c5.number_format = "0.0"

        c6 = ws3.cell(row=row, column=6, value=round(est_total, 1))
        c6.font = Font(name="Arial", size=11)
        c6.alignment = Alignment(horizontal="center")
        c6.number_format = "0.0"

        c7 = ws3.cell(row=row, column=7, value=round(est_rtf, 2))
        c7.font = Font(name="Arial", size=11)
        c7.alignment = Alignment(horizontal="center")
        c7.number_format = "0.00"

        if dev["is_actual"]:
            for col in range(1, len(headers4) + 1):
                ws3.cell(row=row, column=col).fill = actual_fill

    apply_borders(ws3, r, r + len(devices), 1, len(headers4))
    alt_row_fill(ws3, r + 2, r + len(devices), 1, len(headers4))
    # Re-apply actual highlight (after alt_row override)
    for i, dev in enumerate(devices):
        if dev["is_actual"]:
            for col in range(1, len(headers4) + 1):
                ws3.cell(row=r + 1 + i, column=col).fill = actual_fill

    # Notes section
    r_notes = r + len(devices) + 2
    style_title(ws3, r_notes, 1, "Ghi chú ước lượng", 7)
    notes = [
        "• Detect stage chiếm ~95% tổng thời gian pipeline, là bottleneck chính (GPU-bound inference).",
        "• Smooth stage (CPU post-processing) rất nhanh (< 0.1s), không đáng kể.",
        "• Render stage phụ thuộc CPU + ffmpeg encoder, có thể tăng tốc với --nvenc (GPU encode).",
        "• Máy văn phòng (CPU only): PaddleOCR chạy trên CPU chậm ~10x so với GPU → khuyến nghị dùng GPU.",
        "• Apple Silicon: PaddleOCR hỗ trợ MPS backend nhưng chưa tối ưu bằng CUDA, hiệu năng ước lượng thận trọng.",
        "• RTF < 1.0 nghĩa là xử lý nhanh hơn thời gian thực (real-time capable).",
        f"• Video test: 720x1280, {duration_s:.0f}s, {n_frames} frames @ {fps_video:.0f} fps.",
        "• Các giá trị ước lượng dựa trên TFLOPS tương đối và kinh nghiệm benchmark PaddleOCR text detection.",
    ]
    for i, note in enumerate(notes):
        row = r_notes + 1 + i
        c = ws3.cell(row=row, column=1, value=note)
        c.font = Font(name="Arial", size=9, color="595959")
        ws3.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)

    auto_width(ws3)
    # Override some column widths for better layout
    ws3.column_dimensions["A"].width = 32
    ws3.column_dimensions["B"].width = 38
    for letter in ["C", "D", "E", "F", "G"]:
        ws3.column_dimensions[letter].width = 18

    # =========================================================================
    # Save
    # =========================================================================
    wb.save(output_path)
    print(f"[report] Saved to: {os.path.abspath(output_path)}")
    print(f"         Sheets: {len(wb.worksheets)} sheets created")


if __name__ == "__main__":
    timing_path = os.path.join("work", "AI Engineer test.timing.json")
    output_path = "Performance_Report_Subtitle_Detection.xlsx"
    create_report(timing_path, output_path)
