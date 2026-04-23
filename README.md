# Monitoring-a-urban-site-using-a-video-stream

A real-time video analysis app built with **Streamlit**, **YOLO 26s**, and **OpenCV**. It detects and tracks persons and vehicles in video footage, with configurable crowd alerts and ROI (Region of Interest) monitoring.

---

## Features

- **YOLO 26s object tracking** with ByteTrack for persistent ID assignment
- **KNN background subtraction** combined with a static median background to filter out stationary objects — only moving detections are counted
- **Global crowd alert** — triggers when the number of persons in frame exceeds a configurable limit
- **ROI monitoring** — define a sub-region of the frame to monitor person count and motion density independently
- **Unique object counting** — tracks unique persons and vehicles across the entire video (with a minimum frame persistence filter to avoid false counts)
- **Live stats panel** with per-frame and cumulative metrics
- **Optional motion mask overlay** to visualize the KNN foreground mask

---

## Requirements

- Python 3.9+
- A YOLO model weight file (e.g. `yolo26s.pt` or any standard `yolov8*.pt`)
- GPU optional but recommended for real-time performance

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
streamlit run main.py
```

Then open the app in your browser (default: `http://localhost:8501`).

### Providing video input

In the sidebar, choose one of two input modes:

- **Local path** : Paste the full path to a video file on disk. This avoids re-uploading large files and is faster.
- **Upload file** *(recommended in Streamlit app)*: Upload an `.mp4`, `.avi`, or `.mov` file directly through the browser.

---

## Sidebar Parameters

| Section | Parameter | Description |
|---|---|---|
| **Parameters** | YOLO model path | Path to your `.pt` weight file |
| | Confidence threshold | Minimum detection confidence (0–1) |
| | Motion ratio threshold | Minimum foreground pixel ratio inside a bounding box for the detection to be kept |
| | Show motion mask | Toggle the KNN foreground mask display |
| | Frame skip | Skip N frames between processed frames to increase speed |
| **YOLO inference** | imgsz | Inference image size |
| | NMS IoU | Non-maximum suppression IoU threshold |
| | FP16 half precision | Enable half-precision for GPU inference |
| **Alert Settings** | Global person count alert | Enable/disable the global crowd alert |
| | Max persons in frame | Person count threshold to trigger the alert |
| | Enable ROI monitoring | Toggle ROI-based alerts |
| | ROI coordinates | Define ROI as normalized (0–1) x1, y1, x2, y2 values |
| | Max persons in ROI | Person count threshold within the ROI |
| | Max motion pixel ratio in ROI | Foreground pixel density threshold within the ROI |

---

## Detected Classes

Only the following COCO classes are tracked:

| Class ID | Label |
|---|---|
| 0 | Person |
| 2 | Car |
| 5 | Bus |
| 7 | Truck |

---

## How It Works

1. **Static median background** is computed from the first 60 frames and used to warm up the KNN subtractor before processing begins.
2. For each frame, the KNN subtractor produces a foreground mask. Detections are only kept if the foreground pixel ratio within their bounding box exceeds the motion threshold — this eliminates parked vehicles and stationary people.
3. ByteTrack assigns persistent IDs across frames. An object is added to the unique count only after it has been continuously tracked for **15 or more frames**, reducing false positives from noise.
4. Alerts are evaluated per frame and displayed as a banner overlay in the UI. The alert state is cleared at the end of each frame to avoid stale alerts persisting on screen.

---

## Notes

- The default model path is `yolo26s.pt`. Change this in the sidebar to point to any compatible YOLO weights file.
- FP16 half precision only works on CUDA-enabled GPUs; leave it off for CPU inference.
- Uploaded files are written to a temporary location and deleted automatically after processing.
