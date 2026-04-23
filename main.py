import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# --- Page Config ---
st.set_page_config(
    page_title="Video Detection (Optimized KNN)",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ALLOWED_CLASSES = {0, 2, 5, 7}
VEHICLE_CLASSES = {2, 5, 7}


class KNNBackgroundModel:
    def __init__(self, history=500, threshold=400, detect_shadows=True):
        self.subtractor = cv2.createBackgroundSubtractorKNN(
            history=history, dist2Threshold=threshold, detectShadows=detect_shadows
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def apply(self, frame):
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        mask = self.subtractor.apply(blur)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        return mask


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


def compute_static_median(video_path, num_frames=60):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def draw_roi_on_frame(frame, roi_coords, color=(0, 255, 100), thickness=2):
    h, w = frame.shape[:2]
    x1 = int(roi_coords[0] * w)
    y1 = int(roi_coords[1] * h)
    x2 = int(roi_coords[2] * w)
    y2 = int(roi_coords[3] * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, "ROI", (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame, (x1, y1, x2, y2)


def count_persons_in_roi(boxes, ids, clss, roi_px):
    """Count persons whose center point falls within the ROI rectangle."""
    rx1, ry1, rx2, ry2 = roi_px
    count = 0
    roi_ids = set()
    for i, box in enumerate(boxes):
        if int(clss[i]) != 0:
            continue
        # Use bottom-center point (feet) for more accurate ROI membership
        cx = int((box[0] + box[2]) / 2)
        cy = int(box[3])  # bottom edge (feet) instead of center
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            count += 1
            roi_ids.add(ids[i])
    return count, roi_ids


def compute_roi_motion_ratio(fg_mask, roi_px):
    rx1, ry1, rx2, ry2 = roi_px
    roi_mask = fg_mask[ry1:ry2, rx1:rx2]
    if roi_mask.size == 0:
        return 0.0
    return float((roi_mask > 0).sum()) / roi_mask.size


def render_alert_banner(alerts: list):
    """Render alert banner. Returns True if rendered."""
    if not alerts:
        return False
    # FIX: corrected markdown bold syntax (no spaces inside **)
    items_html = "".join(f"<li>⚠️ {a}</li>" for a in alerts)
    html = f"""
    <div style="
        background: linear-gradient(135deg, #ff4b4b 0%, #c0392b 100%);
        border: 2px solid #ff0000;
        border-radius: 10px;
        padding: 14px 20px;
        margin: 8px 0 12px 0;
        box-shadow: 0 0 16px 4px rgba(255,0,0,0.45);
    ">
        <span style="font-size:1.25rem; font-weight:700; color:#fff; letter-spacing:1px;">
            🚨 CROWD ALERT
        </span>
        <ul style="margin:6px 0 0 0; padding-left:20px; color:#fff; font-size:1rem; line-height:1.7;">
            {items_html}
        </ul>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    return True


def main():
    st.title("🎬 Smart Video Detection System")
    st.markdown("Optimization: **Static Median + KNN Subtractor + ID Persistence** | 🚨 **Crowd Warning + ROI Monitoring**")

    with st.sidebar:
        st.header("Parameters")
        model_path = st.text_input("YOLO model path", value="yolo26s.pt")
        conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
        motion_threshold = st.slider("Motion ratio threshold", 0.01, 0.15, 0.05, 0.01)
        show_fgmask = st.checkbox("Show motion mask", value=True)
        skip_frames = st.slider("Frame skip (speed)", 0, 5, 0)

        st.divider()
        st.header("YOLO inference")
        imgsz = st.select_slider("imgsz", options=[320, 416, 480, 640, 736, 832, 1024, 1280], value=640)
        iou_threshold = st.slider("NMS IoU", 0.1, 0.9, 0.45, 0.05)
        half_precision = st.checkbox("FP16 half precision (GPU only)", value=False)

        st.divider()
        st.header("🚨 Alert Settings")
        enable_global_alert = st.checkbox("Enable global person count alert", value=True)
        global_person_limit = st.number_input("Max persons in frame", min_value=1, max_value=200, value=10, step=1)

        st.markdown("**ROI Region (ratio 0.0–1.0)**")
        enable_roi = st.checkbox("Enable ROI monitoring", value=False)
        col_r1, col_r2 = st.columns(2)
        roi_x1 = col_r1.number_input("ROI x1", 0.0, 1.0, 0.25, 0.05)
        roi_y1 = col_r2.number_input("ROI y1", 0.0, 1.0, 0.25, 0.05)
        roi_x2 = col_r1.number_input("ROI x2", 0.0, 1.0, 0.75, 0.05)
        roi_y2 = col_r2.number_input("ROI y2", 0.0, 1.0, 0.75, 0.05)
        roi_person_limit = st.number_input("Max persons in ROI", min_value=1, max_value=100, value=5, step=1)
        roi_motion_limit = st.slider("Max motion pixel ratio in ROI", 0.05, 1.0, 0.30, 0.05)
        roi_coords = (roi_x1, roi_y1, roi_x2, roi_y2)

        st.divider()
        st.header("Video input")
        input_mode = st.radio("Input mode", ["Local path (recommended)", "Upload file"], horizontal=True)
        video_path = None
        need_cleanup = False

        if input_mode == "Local path (recommended)":
            path_input = st.text_input("Full video file path", placeholder=r"C:\path\to\video.mp4")
            if path_input and Path(path_input.strip()).exists():
                video_path = str(Path(path_input.strip()).resolve())
        else:
            uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_file.read())
                    video_path = tmp.name
                    need_cleanup = True

    if not video_path:
        st.info("Select a local path or upload a video to start.")
        return

    try:
        model = load_model(model_path)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        return

    bg_engine = KNNBackgroundModel()

    with st.status("Computing static background...") as status:
        median_bg = compute_static_median(video_path)
        if median_bg is not None:
            for _ in range(10):
                bg_engine.apply(median_bg)
        status.update(label="Model Ready!", state="complete")

    # FIX: Create placeholder OUTSIDE the loop so it persists across frames
    alert_placeholder = st.empty()
    col_video, col_stats = st.columns([3, 1])
    video_placeholder = col_video.empty()
    fg_placeholder = col_video.empty() if show_fgmask else None
    stats_placeholder = col_stats.empty()
    progress_bar = st.progress(0)

    cbtn1, cbtn2 = st.columns(2)
    run_btn = cbtn1.button("▶️ Start", type="primary")
    if "stop" not in st.session_state:
        st.session_state.stop = False
    if cbtn2.button("⏹️ Stop"):
        st.session_state.stop = True

    if run_btn:
        st.session_state.stop = False
        seen_persons = set()
        seen_vehicles = set()
        id_persistence = {}
        frame_idx = 0
        start_time = time.time()

        while cap.isOpened() and not st.session_state.stop:
            ret, frame = cap.read()
            if not ret:
                break

            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            fg_mask = bg_engine.apply(frame)

            results = model.track(
                frame, persist=True, tracker="bytetrack.yaml", verbose=False,
                imgsz=imgsz, conf=conf_threshold, iou=iou_threshold, half=half_precision
            )[0]

            curr_person_ids = set()
            curr_vehicle_ids = set()
            raw_boxes, raw_ids, raw_clss = [], [], []

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids   = results.boxes.id.cpu().numpy().astype(int)
                clss  = results.boxes.cls.cpu().numpy().astype(int)

                keep_indices = []
                for i, box in enumerate(boxes):
                    cls_id = int(clss[i])

                    if cls_id not in ALLOWED_CLASSES:
                        continue

                    x1, y1, x2, y2 = box
                    roi_region = fg_mask[max(0, y1):y2, max(0, x1):x2]
                    if roi_region.size == 0 or (roi_region > 0).mean() <= motion_threshold:
                        continue

                    keep_indices.append(i)
                    obj_id = int(ids[i])

                    id_persistence[obj_id] = id_persistence.get(obj_id, 0) + 1
                    if id_persistence[obj_id] >= 15:
                        if cls_id == 0:
                            seen_persons.add(obj_id)
                        elif cls_id in VEHICLE_CLASSES:
                            seen_vehicles.add(obj_id)

                    if cls_id == 0:
                        curr_person_ids.add(obj_id)
                    elif cls_id in VEHICLE_CLASSES:
                        curr_vehicle_ids.add(obj_id)

                    raw_boxes.append(box)
                    raw_ids.append(obj_id)
                    raw_clss.append(cls_id)

                results.boxes = results.boxes[keep_indices]

            # --- Alert logic ---
            alerts = []
            global_person_count = len(curr_person_ids)

            if enable_global_alert and global_person_count >= global_person_limit:
                # FIX: corrected bold markdown syntax (no spaces inside **)
                alerts.append(f"Global person count <b>{global_person_count}</b> ≥ limit <b>{int(global_person_limit)}</b>")

            roi_person_count = 0
            roi_motion_ratio = 0.0
            roi_px = None

            if enable_roi:
                rx1 = int(roi_coords[0] * w)
                ry1 = int(roi_coords[1] * h)
                rx2 = int(roi_coords[2] * w)
                ry2 = int(roi_coords[3] * h)
                roi_px = (rx1, ry1, rx2, ry2)

                if raw_boxes:
                    roi_person_count, _ = count_persons_in_roi(raw_boxes, raw_ids, raw_clss, roi_px)
                roi_motion_ratio = compute_roi_motion_ratio(fg_mask, roi_px)

                if roi_person_count >= roi_person_limit:
                    # FIX: corrected bold markdown syntax
                    alerts.append(f"ROI person count <b>{roi_person_count}</b> ≥ limit <b>{int(roi_person_limit)}</b>")
                if roi_motion_ratio >= roi_motion_limit:
                    alerts.append(f"ROI motion density <b>{roi_motion_ratio:.1%}</b> ≥ limit <b>{roi_motion_limit:.0%}</b>")

            # --- Annotate frame ---
            annotated = results.plot()

            if enable_roi and roi_px:
                roi_color = (0, 0, 255) if alerts else (0, 255, 100)
                annotated, _ = draw_roi_on_frame(annotated, roi_coords, color=roi_color)

            if enable_global_alert and global_person_count >= global_person_limit:
                txt = f"! PERSONS: {global_person_count}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
                cv2.rectangle(annotated, (w - tw - 16, 8), (w - 4, th + 20), (0, 0, 200), -1)
                cv2.putText(annotated, txt, (w - tw - 10, th + 12),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

            # FIX: Always clear the alert placeholder first, then conditionally render
            # This prevents stale alerts from previous frames persisting on screen
            alert_placeholder.empty()
            if alerts:
                with alert_placeholder.container():
                    render_alert_banner(alerts)

            video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            if show_fgmask and fg_placeholder:
                fg_placeholder.image(fg_mask, caption="Motion Mask (KNN)", use_container_width=True)

            with stats_placeholder.container():
                st.markdown("**Current Frame (Moving)**")
                st.metric(
                    "Persons",
                    global_person_count,
                    # FIX: delta now correctly reflects the CURRENT frame's alert state
                    delta="⚠️ ALERT" if (enable_global_alert and global_person_count >= global_person_limit) else None,
                    delta_color="inverse"
                )
                st.metric("Vehicles", len(curr_vehicle_ids))

                if enable_roi:
                    st.divider()
                    st.markdown("**ROI Stats**")
                    st.metric(
                        "ROI Persons",
                        roi_person_count,
                        # FIX: delta also reflects current frame value
                        delta="⚠️" if roi_person_count >= roi_person_limit else None,
                        delta_color="inverse"
                    )
                    st.metric(
                        "ROI Motion",
                        f"{roi_motion_ratio:.1%}",
                        delta="⚠️" if roi_motion_ratio >= roi_motion_limit else None,
                        delta_color="inverse"
                    )

                st.divider()
                st.markdown("Unique Totals (Filtered)")
                st.metric("Unique Persons", len(seen_persons))
                st.metric("Unique Vehicles", len(seen_vehicles))
                st.metric("Elapsed", f"{time.time() - start_time:.1f}s")

            if total_frames > 0:
                progress_bar.progress(min((frame_idx + 1) / total_frames, 1.0))

            frame_idx += 1

        cap.release()
        if need_cleanup and video_path:
            Path(video_path).unlink(missing_ok=True)

        # FIX: Clear alert after video ends so no stale alert lingers
        alert_placeholder.empty()
        st.success("✅ Analysis Finished.")


if __name__ == "__main__":
    main()