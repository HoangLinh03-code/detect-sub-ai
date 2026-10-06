# Video Subtitle Detection & Translation (Detect Sub)

Dự án này cung cấp một pipeline hoàn chỉnh giúp tự động nhận diện phụ đề tiếng Trung (cứng/hardsub) trong video, dịch sang tiếng Việt bằng Google Gemini AI, sau đó xuất ra file `.srt` và render một video mới với phụ đề tiếng Việt.

## 🚀 Các tính năng chính
- **Nhận diện vùng phụ đề (Detection):** Sử dụng PaddleOCR để tìm các vùng chứa chữ trong video, giới hạn trong khu vực (ROI) cụ thể để tối ưu tốc độ.
- **Làm mượt (Smoothing):** Theo dõi (tracking) và hợp nhất (merging) các vùng chữ qua các khung hình để tạo thành các đoạn (segments) phụ đề ổn định.
- **Trích xuất văn bản (OCR):** Đọc chữ tiếng Trung từ các vùng đã nhận diện.
- **Dịch thuật AI (Translation):** Sử dụng API của Google Gemini (hỗ trợ các model như `gemini-3.5-flash-lite`, `gemini-2.0-flash`) để dịch thuật mượt mà, ngữ cảnh tự nhiên.
- **Xuất phụ đề (Export & Render):** Tự động tạo file `.srt` để tiện dùng trong phần mềm dựng phim, hoặc render trực tiếp (burn) phụ đề tiếng Việt lên video mới bằng `ffmpeg` hoặc OpenCV.

## 📁 Cấu trúc thư mục

- `translate.py`: **Script chính** của dự án. Chạy toàn bộ quy trình nhận diện -> dịch -> xuất video & srt.
- `sub_boxes.py`: Script dùng để test/debug quy trình nhận diện bounding box phụ đề (chỉ tìm khung, không đọc chữ/dịch).
- `generate_report.py`: Script tạo báo cáo hiệu năng (Performance Report) dưới dạng file Excel từ file timing `.json`.
- `debug_render.py`: File module chứa các hàm xử lý nền cho OCR, render font và utils (được `translate.py` kế thừa).
- `.env-example`: File mẫu cấu hình môi trường. (Bạn cần copy thành `.env` để sử dụng).

---

# Hướng Dẫn Cài Đặt và Sử Dụng

Tài liệu này hướng dẫn chi tiết cách thiết lập môi trường và chạy các script trong dự án **Detect Sub**.

## Bước 1: Yêu cầu & Cài đặt thư viện (Requirements)

Đảm bảo bạn đã cài đặt Python (khuyên dùng Python 3.8 - 3.10). Cần cài đặt `ffmpeg` trên hệ thống hoặc sử dụng package `imageio-ffmpeg` để xử lý video và âm thanh hiệu quả.

Mở terminal/command prompt và chạy các lệnh sau để cài đặt các thư viện cần thiết:

```bash
# Cài đặt thư viện cho AI, xử lý ảnh và video
pip install paddlepaddle paddleocr opencv-python numpy Pillow

# Cài đặt thư viện cho Gemini AI và biến môi trường
pip install google-genai python-dotenv

# Cài đặt thư viện hỗ trợ render video (kèm ffmpeg) và tạo báo cáo excel
pip install imageio-ffmpeg openpyxl
```

*(Lưu ý: Nếu bạn có GPU NVIDIA và muốn chạy PaddleOCR nhanh hơn, hãy cài đặt bản `paddlepaddle-gpu` phù hợp với phiên bản CUDA của máy bạn theo hướng dẫn trên trang chủ PaddlePaddle).*

## Bước 2: Cấu hình môi trường (.env)

Dự án sử dụng file `.env` để lưu trữ API Key và cấu hình hệ thống, tránh việc lộ API Key khi đưa code lên mạng.

1. Trong thư mục dự án, copy file `.env-example` và đổi tên thành `.env`.
2. Mở file `.env` vừa tạo và chỉnh sửa các thông số:

```ini
# Lấy API Key từ Google AI Studio (https://aistudio.google.com/)
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE

# Chọn model Gemini (ví dụ: gemini-3.5-flash-lite, gemini-2.0-flash)
GEMINI_MODEL=gemini-3.5-flash-lite

# Thiết lập thiết bị chạy OCR (gpu hoặc cpu)
OCR_DEVICE=cpu # hoặc gpu:0
```

## Bước 3: Sử dụng Script Chính (`translate.py`)

`translate.py` là script mạnh nhất để chạy full pipeline: **Nhận diện → Smooth → OCR → Dịch thuật → Render**.

**Cú pháp cơ bản:**
```bash
python translate.py "tên_video_gốc.mp4" -o "tên_video_đầu_ra.mp4"
```

**Ví dụ thực tế:**
```bash
python translate.py "AI Engineer test.mp4" -o "output_vi.mp4"
```
*Kết quả: Bạn sẽ nhận được file `output_vi.mp4` (Video đã được gắn phụ đề dịch) và `output_vi.srt` (File subtitle rời).*

**Các tùy chọn nâng cao:**
- **Chỉ dịch và xuất SRT (Không render video cứng):**
  ```bash
  python translate.py "video.mp4" -o "out.mp4" --no-render
  ```
- **Sử dụng cache, bỏ qua OCR (khi đã nhận diện trước đó, chỉ muốn dịch lại):**
  ```bash
  python translate.py "video.mp4" -o "out.mp4" --no-ocr
  ```
- **Chỉ chạy các bước cụ thể:**
  ```bash
  python translate.py "video.mp4" --stages smooth,render
  ```

## Bước 4: Debug và Test Bounding Box (`sub_boxes.py`)

Nếu bạn chỉ muốn kiểm tra xem AI nhận diện khung phụ đề (bounding box) có chuẩn không mà chưa cần dịch:

```bash
# Chạy full các bước để vẽ khung xanh bao quanh phụ đề
python sub_boxes.py "video.mp4" -o "video_boxed.mp4"

# Chạy bằng CPU
python sub_boxes.py "video.mp4" -o "video_cpu.mp4" --device cpu
```
Video đầu ra sẽ hiển thị các ô vuông xanh bao quanh chữ để bạn kiểm chứng thông số ROI (khu vực nhận diện).

## Bước 5: Tạo báo cáo hiệu năng (`generate_report.py`)

Sau khi chạy xong các script, dự án sẽ sinh ra các file `.timing.json` trong thư mục `work/`. Bạn có thể tạo bảng báo cáo dạng Excel để xem tốc độ xử lý:

```bash
python generate_report.py
```
Kết quả sẽ xuất ra file `Performance_Report_Subtitle_Detection.xlsx` chứa chi tiết thông số FPS, thời gian xử lý các bước.

## ⚠️ Xử lý sự cố thường gặp (Troubleshooting)
- **Lỗi thiếu FFmpeg:** Script không render được video? Cài đặt thêm thư viện: `pip install imageio-ffmpeg`
- **Lỗi Gemini API (Rate limit):** Do API free có giới hạn số lượt request/phút. Bạn có thể tăng tham số `TRANSLATE_DELAY` trong file `.env` (ví dụ lên `15.0`) để giãn cách thời gian gọi API.
- **Không nhận diện được chữ:** Chỉnh lại `OCR_ROI_TOP` trong `.env`. Mặc định `0.55` nghĩa là chỉ quét nửa dưới của video. Nếu phụ đề nằm cao hơn, hãy giảm số này (ví dụ: `0.3`).
