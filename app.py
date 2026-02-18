# app.py
import os
import io
import html
import math
import time
import hashlib
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern

# Clustering (optional section in app; used for "cluster number" tooltips)
try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Multi-Video Frame Similarity", layout="wide")

LBP_P = 16
LBP_R = 2
LBP_METHOD = "uniform"

# ----------------------------
# Labeling helpers (Option 3: hover-only metadata)
# ----------------------------
def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def video_letter_name(video_index: int) -> str:
    return f"Video {chr(65 + int(video_index))}"

def build_hover_tooltip(
    r,
    similarity_score: Optional[float] = None,
    cluster_id: Optional[int] = None,
) -> str:
    parts = []
    parts.append(f"Frame ID: {r.frame_id}")
    parts.append(f"Exact time: {r.t_sec:.6f} seconds")
    parts.append(f"Frame index: {r.frame_index}")
    parts.append(f"Entropy: {r.entropy:.6f}")
    parts.append(f"Edge density: {r.edge:.6f}")
    parts.append(f"Brightness: {r.brightness:.2f}")

    if similarity_score is not None:
        parts.append(f"Similarity score: {float(similarity_score):.6f}")

    if cluster_id is not None:
        parts.append(f"Cluster: {int(cluster_id)}")

    # You can add more fields here if you like (e.g., video filename)
    parts.append(f"Source file: {r.video_name}")

    return "\n".join(parts)

def render_frame_label(
    r,
    similarity_score: Optional[float] = None,
    cluster_id: Optional[int] = None,
) -> None:
    """
    Visible:  Video A · 00:12
    Hover:    frame ID, exact timestamp, entropy, similarity score, cluster number, etc.
    """
    visible = f"{video_letter_name(r.video_index)} · {format_time(r.t_sec)}"
    tooltip = build_hover_tooltip(r, similarity_score=similarity_score, cluster_id=cluster_id)

    visible_e = html.escape(visible)
    tooltip_e = html.escape(tooltip)

    st.markdown(
        f"""
        <div title="{tooltip_e}"
             style="
                font-size:13px;
                margin-top:2px;
                white-space:nowrap;
                overflow:hidden;
                text-overflow:ellipsis;
                cursor:default;
                opacity:0.92;
             ">
          {visible_e}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# Image + descriptor helpers
# ----------------------------
def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def _resize_keep_aspect(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / float(max(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def shannon_entropy_gray(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 100, 200)
    return float(edges.mean() / 255.0)  # 0..1

def brightness_mean(gray: np.ndarray) -> float:
    return float(gray.mean())  # 0..255

def color_hist_bgr(img_bgr: np.ndarray, bins: int = 32) -> np.ndarray:
    hist = []
    for ch in range(3):
        h = cv2.calcHist([img_bgr], [ch], None, [bins], [0, 256]).ravel()
        h = h / (h.sum() + 1e-12)
        hist.append(h)
    return np.concatenate(hist).astype(np.float32)

def hue_hist(img_bgr: np.ndarray, bins: int = 36) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]  # 0..179
    h = cv2.calcHist([hue], [0], None, [bins], [0, 180]).ravel()
    h = h / (h.sum() + 1e-12)
    return h.astype(np.float32)

def lbp_hist(gray: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method=LBP_METHOD)
    n_bins = LBP_P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=False)
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-12)
    return hist

def corr_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    cos = float(np.dot(a, b) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    return (cos + 1.0) / 2.0

def hist_corr_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel().reshape(-1, 1)
    b = b.astype(np.float32).ravel().reshape(-1, 1)
    c = float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
    c = max(-1.0, min(1.0, c))
    return (c + 1.0) / 2.0

# ----------------------------
# Data structure
# ----------------------------
@dataclass
class FrameRecord:
    frame_id: str
    video_name: str
    video_index: int
    t_sec: float
    frame_index: int
    thumb_path: str

    # descriptors
    gray_small: np.ndarray
    color_hist: np.ndarray
    hue_hist: np.ndarray
    entropy: float
    edge: float
    texture_hist: np.ndarray
    brightness: float

# ----------------------------
# Similarity computation
# ----------------------------
def compute_pair_similarity(
    A: FrameRecord,
    B: FrameRecord,
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    # Structural alignment: SSIM on small grayscale
    try:
        s = float(ssim(A.gray_small, B.gray_small, data_range=255))
        s = max(-1.0, min(1.0, s))
        ssim_sim = (s + 1.0) / 2.0
    except Exception:
        ssim_sim = 0.0

    color_sim = hist_corr_sim(A.color_hist, B.color_hist)

    e_diff = abs(A.entropy - B.entropy)
    ent_sim = 1.0 - min(1.0, e_diff / 8.0)

    edge_sim = 1.0 - min(1.0, abs(A.edge - B.edge))

    tex_sim = corr_sim(A.texture_hist, B.texture_hist)

    bri_sim = 1.0 - min(1.0, abs(A.brightness - B.brightness) / 255.0)

    hue_sim = hist_corr_sim(A.hue_hist, B.hue_hist)

    comps = {
        "structural_alignment": ssim_sim,
        "color_histogram": color_sim,
        "entropy_similarity": ent_sim,
        "edge_complexity": edge_sim,
        "texture_correlation": tex_sim,
        "brightness_similarity": bri_sim,
        "hue_distribution": hue_sim,
    }

    wsum = sum(max(0.0, float(weights.get(k, 0.0))) for k in comps.keys()) + 1e-12
    score = sum(comps[k] * max(0.0, float(weights.get(k, 0.0))) for k in comps.keys()) / wsum
    return float(score), comps

# ----------------------------
# Video -> frames
# ----------------------------
def extract_frames(
    video_path: str,
    video_name: str,
    video_index: int,
    mode: str,
    interval_value: float,
    max_frames: int,
    thumb_max_side: int,
    analysis_max_side: int,
    thumbs_dir: str,
) -> List[FrameRecord]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if mode == "Every N seconds":
        step = max(1, int(round(interval_value * fps)))
    else:
        step = max(1, int(round(interval_value)))

    frames: List[FrameRecord] = []
    frame_idx = 0
    grabbed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            t_sec = frame_idx / fps

            # thumbnail
            thumb = _resize_keep_aspect(frame, thumb_max_side)
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            thumb_img = Image.fromarray(thumb_rgb)

            frame_id = f"v{video_index}_f{frame_idx}"
            thumb_path = os.path.join(thumbs_dir, f"{frame_id}.jpg")
            thumb_img.save(thumb_path, quality=90)

            # analysis frame
            analysis_bgr = _resize_keep_aspect(frame, analysis_max_side)
            gray = _to_gray(analysis_bgr).astype(np.uint8)

            rec = FrameRecord(
                frame_id=frame_id,
                video_name=video_name,
                video_index=video_index,
                t_sec=float(t_sec),
                frame_index=int(frame_idx),
                thumb_path=thumb_path,
                gray_small=gray,
                color_hist=color_hist_bgr(analysis_bgr, bins=32),
                hue_hist=hue_hist(analysis_bgr, bins=36),
                entropy=shannon_entropy_gray(gray),
                edge=edge_density(gray),
                texture_hist=lbp_hist(gray),
                brightness=brightness_mean(gray),
            )
            frames.append(rec)
            grabbed += 1
            if grabbed >= max_frames:
                break

        frame_idx += 1
        if total_frames and frame_idx >= total_frames:
            break

    cap.release()
    return frames

def write_bytes_to_temp(bts: bytes, name: str) -> str:
    suffix = os.path.splitext(name)[1] or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(bts)
    return path

def build_cache_key(
    file_bytes_and_names: List[Tuple[bytes, str]],
    mode: str,
    interval_value: float,
    max_frames: int,
    thumb_max_side: int,
    analysis_max_side: int,
) -> str:
    h = hashlib.sha256()
    h.update(mode.encode("utf-8"))
    h.update(str(interval_value).encode("utf-8"))
    h.update(str(max_frames).encode("utf-8"))
    h.update(str(thumb_max_side).encode("utf-8"))
    h.update(str(analysis_max_side).encode("utf-8"))
    # include file content + names
    for bts, name in file_bytes_and_names:
        h.update(name.encode("utf-8"))
        h.update(hashlib.sha256(bts).digest())
    return h.hexdigest()

def process_videos(
    file_bytes_and_names: List[Tuple[bytes, str]],
    mode: str,
    interval_value: float,
    max_frames: int,
    thumb_max_side: int,
    analysis_max_side: int,
) -> Tuple[List[FrameRecord], str]:
    thumbs_dir = tempfile.mkdtemp(prefix="thumbs_")
    temp_paths = []
    all_records: List[FrameRecord] = []

    try:
        for i, (bts, name) in enumerate(file_bytes_and_names):
            path = write_bytes_to_temp(bts, name)
            temp_paths.append(path)

            recs = extract_frames(
                video_path=path,
                video_name=name,
                video_index=i,
                mode=mode,
                interval_value=interval_value,
                max_frames=max_frames,
                thumb_max_side=thumb_max_side,
                analysis_max_side=analysis_max_side,
                thumbs_dir=thumbs_dir,
            )
            all_records.extend(recs)

        return all_records, thumbs_dir
    finally:
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

# ----------------------------
# Clustering (scene clustering across all videos)
# ----------------------------
def build_feature_matrix_for_clustering(records: List[FrameRecord]) -> np.ndarray:
    # Use descriptors that travel well across videos
    feats = []
    for r in records:
        vec = np.concatenate(
            [
                r.color_hist,
                r.hue_hist,
                r.texture_hist,
                np.array(
                    [
                        r.entropy / 8.0,
                        r.edge,
                        r.brightness / 255.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        feats.append(vec)
    X = np.vstack(feats).astype(np.float32)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs

def cluster_frames(records: List[FrameRecord], distance_threshold: float) -> np.ndarray:
    X = build_feature_matrix_for_clustering(records)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(distance_threshold),
        linkage="ward",
    )
    labels = clustering.fit_predict(X)
    return labels

# ----------------------------
# UI
# ----------------------------
st.title("Multi-Video Frame Similarity Analyzer")

with st.sidebar:
    st.header("1) Upload videos")
    uploads = st.file_uploader(
        "Upload videos (3 or more recommended).",
        type=["mp4", "mov", "m4v", "avi", "mkv", "webm"],
        accept_multiple_files=True,
    )

    st.header("2) Frame sampling")
    mode = st.radio("Sampling mode", ["Every N seconds", "Every N frames"], index=0)
    if mode == "Every N seconds":
        interval_value = st.number_input("Interval (seconds)", min_value=0.1, value=2.0, step=0.1)
    else:
        interval_value = st.number_input("Interval (frames)", min_value=1.0, value=30.0, step=1.0)

    st.header("3) Limits & performance")
    max_frames = st.number_input("Max frames per video", min_value=10, max_value=5000, value=250, step=10)
    thumb_max_side = st.slider("Thumbnail max side (px)", min_value=120, max_value=640, value=220, step=10)
    analysis_max_side = st.slider("Analysis max side (px)", min_value=120, max_value=720, value=320, step=20)

    st.header("4) Metric weights")
    st.caption("Weights control how each metric contributes to the combined similarity score.")
    weights = {
        "structural_alignment": st.slider("Structural alignment (SSIM)", 0.0, 3.0, 1.0, 0.1),
        "color_histogram": st.slider("Color histogram", 0.0, 3.0, 1.0, 0.1),
        "entropy_similarity": st.slider("Entropy similarity", 0.0, 3.0, 1.0, 0.1),
        "edge_complexity": st.slider("Edge complexity", 0.0, 3.0, 1.0, 0.1),
        "texture_correlation": st.slider("Texture correlation (LBP)", 0.0, 3.0, 1.0, 0.1),
        "brightness_similarity": st.slider("Brightness similarity", 0.0, 3.0, 1.0, 0.1),
        "hue_distribution": st.slider("Hue distribution", 0.0, 3.0, 1.0, 0.1),
    }

    st.header("5) Retrieval")
    top_k = st.slider("Top-K matches to show", min_value=3, max_value=50, value=12, step=1)

if not uploads:
    st.info("Upload your videos in the sidebar to begin.")
    st.stop()

if len(uploads) < 2:
    st.warning("Upload at least 2 videos to enable cross-video comparison.")
    st.stop()

file_bytes_and_names = [(u.getbuffer().tobytes(), u.name) for u in uploads]

cache_key = build_cache_key(
    file_bytes_and_names=file_bytes_and_names,
    mode=mode,
    interval_value=float(interval_value),
    max_frames=int(max_frames),
    thumb_max_side=int(thumb_max_side),
    analysis_max_side=int(analysis_max_side),
)

if "cache_key" not in st.session_state or st.session_state.cache_key != cache_key:
    with st.spinner("Extracting frames and computing descriptors..."):
        records, thumbs_dir = process_videos(
            file_bytes_and_names=file_bytes_and_names,
            mode=mode,
            interval_value=float(interval_value),
            max_frames=int(max_frames),
            thumb_max_side=int(thumb_max_side),
            analysis_max_side=int(analysis_max_side),
        )
    st.session_state.cache_key = cache_key
    st.session_state.records = records
    st.session_state.thumbs_dir = thumbs_dir
    st.session_state.cluster_labels = None  # reset clustering on new extraction
else:
    records = st.session_state.records
    thumbs_dir = st.session_state.thumbs_dir

if not records:
    st.error("No frames were extracted. Try a smaller interval or a higher max-frames-per-video.")
    st.stop()

rec_by_id: Dict[str, FrameRecord] = {r.frame_id: r for r in records}

df = pd.DataFrame(
    [
        {
            "frame_id": r.frame_id,
            "video": r.video_name,
            "video_index": r.video_index,
            "time_sec": round(r.t_sec, 3),
            "frame_index": r.frame_index,
            "entropy": r.entropy,
            "edge_density": r.edge,
            "brightness": r.brightness,
            "thumb_path": r.thumb_path,
        }
        for r in records
    ]
)

# ---------------------------------
# Extracted frames browser
# ---------------------------------
st.subheader("Extracted frames")

c1, c2, c3 = st.columns([1.2, 1.2, 2.6])

with c1:
    video_filter = st.multiselect(
        "Filter by source file",
        options=sorted(df["video"].unique().tolist()),
        default=sorted(df["video"].unique().tolist()),
    )

with c2:
    df_f = df[df["video"].isin(video_filter)].copy()
    df_f = df_f.sort_values(["video_index", "time_sec", "frame_index"]).reset_index(drop=True)
    st.metric("Frames extracted (filtered)", len(df_f))

with c3:
    st.caption("Tip: Pick a query frame below, then retrieve closest matches across *all* uploaded videos.")

# thumbnail grid (limited)
max_grid = 60
grid_df = df_f.head(max_grid)

# If clustering exists, map frame_id -> cluster label
cluster_map = None
if st.session_state.get("cluster_labels") is not None:
    cluster_map = {records[i].frame_id: int(st.session_state.cluster_labels[i]) for i in range(len(records))}

cols = st.columns(6)
for i, row in enumerate(grid_df.itertuples(index=False)):
    col = cols[i % 6]
    with col:
        st.image(row.thumb_path, use_container_width=True)
        r = rec_by_id[row.frame_id]
        cid = cluster_map.get(r.frame_id) if cluster_map else None
        render_frame_label(r, cluster_id=cid)

st.divider()

# ---------------------------------
# Similarity search across videos
# ---------------------------------
st.subheader("Similarity search across all videos")

left, right = st.columns([1.0, 2.0], vertical_alignment="top")

with left:
    query_frame_id = st.selectbox(
        "Choose a query frame",
        options=df_f["frame_id"].tolist(),
        format_func=lambda fid: f"{video_letter_name(rec_by_id[fid].video_index)} · {format_time(rec_by_id[fid].t_sec)}",
    )
    query = rec_by_id[query_frame_id]
    st.image(query.thumb_path, use_container_width=True)
    cid = cluster_map.get(query.frame_id) if cluster_map else None
    render_frame_label(query, cluster_id=cid)

    exclude_same_frame = st.checkbox("Exclude the exact same frame from results", value=True)
    exclude_same_video = st.checkbox("Exclude frames from the same video", value=False)

with right:
    sims = []
    for r in records:
        if exclude_same_frame and r.frame_id == query.frame_id:
            continue
        if exclude_same_video and r.video_index == query.video_index:
            continue
        score, comps = compute_pair_similarity(query, r, weights)
        sims.append(
            {
                "score": score,
                "frame_id": r.frame_id,
                "video": r.video_name,
                "time_sec": r.t_sec,
                "frame_index": r.frame_index,
                **{f"m_{k}": v for k, v in comps.items()},
                "thumb_path": r.thumb_path,
            }
        )

    sims_df = pd.DataFrame(sims).sort_values("score", ascending=False).head(int(top_k)).reset_index(drop=True)

    if sims_df.empty:
        st.warning("No matches found under the current filters.")
    else:
        st.write("Top matches")

        gallery_cols = st.columns(4)
        for i, row in enumerate(sims_df.itertuples(index=False)):
            with gallery_cols[i % 4]:
                st.image(row.thumb_path, use_container_width=True)
                rr = rec_by_id[row.frame_id]
                cid2 = cluster_map.get(rr.frame_id) if cluster_map else None
                # Visible stays clean; score & cluster stay on hover
                render_frame_label(rr, similarity_score=row.score, cluster_id=cid2)

        with st.expander("Show match table (with metric breakdown)"):
            show_cols = [
                "score",
                "video",
                "time_sec",
                "frame_index",
                "frame_id",
                "m_structural_alignment",
                "m_color_histogram",
                "m_entropy_similarity",
                "m_edge_complexity",
                "m_texture_correlation",
                "m_brightness_similarity",
                "m_hue_distribution",
            ]
            st.dataframe(sims_df[show_cols], use_container_width=True)

st.divider()

# ---------------------------------
# Scene clustering across all videos (for cluster numbers in hover)
# ---------------------------------
st.subheader("Scene clustering across all videos")

if not SKLEARN_OK:
    st.warning("Clustering requires scikit-learn. Add `scikit-learn` to requirements.txt to enable this section.")
else:
    st.caption("This groups frames into visually similar “scenes” across *all* uploaded videos.")

    cluster_threshold = st.slider(
        "Cluster sensitivity (lower = more clusters, higher = fewer clusters)",
        min_value=1.0,
        max_value=15.0,
        value=5.0,
        step=0.5,
    )

    cc1, cc2 = st.columns([1, 2])
    with cc1:
        do_cluster = st.button("Cluster frames into scenes")

    with cc2:
        if st.session_state.get("cluster_labels") is not None:
            n_clusters = len(set(map(int, st.session_state.cluster_labels.tolist())))
            st.info(f"Current clustering: {n_clusters} clusters (hover labels now show cluster numbers).")
        else:
            st.info("No clustering computed yet.")

    if do_cluster:
        with st.spinner("Clustering frames..."):
            labels = cluster_frames(records, distance_threshold=float(cluster_threshold))
            st.session_state.cluster_labels = labels

        # refresh cluster_map
        cluster_map = {records[i].frame_id: int(labels[i]) for i in range(len(records))}
        n_clusters = len(set(cluster_map.values()))
        st.success(f"Detected {n_clusters} clusters.")

        # Show a compact preview: one row per cluster (first few frames)
        st.caption("Cluster preview (first ~18 frames per cluster shown)")
        for cluster_id in sorted(set(cluster_map.values())):
            st.markdown(f"### Cluster {cluster_id}")
            cluster_records = [r for r in records if cluster_map.get(r.frame_id) == cluster_id]
            cluster_records = sorted(cluster_records, key=lambda x: (x.video_index, x.t_sec))[:18]

            cols = st.columns(6)
            for i, r in enumerate(cluster_records):
                with cols[i % 6]:
                    st.image(r.thumb_path, use_container_width=True)
                    render_frame_label(r, cluster_id=cluster_id)

st.divider()

# ---------------------------------
# Cross-video similarity summary (optional)
# ---------------------------------
st.subheader("Cross-video similarity summary (optional)")
st.caption(
    "Rough video-to-video similarity: for each frame in Video A, take its best match in Video B, then average."
)

do_summary = st.checkbox("Compute cross-video similarity matrix", value=False)

if do_summary:
    with st.spinner("Computing cross-video similarity matrix... (can be slow)"):
        videos = sorted(df["video"].unique().tolist())
        by_video: Dict[str, List[FrameRecord]] = {v: [] for v in videos}
        for r in records:
            by_video[r.video_name].append(r)

        n = len(videos)
        mat = np.zeros((n, n), dtype=np.float32)

        for i, va in enumerate(videos):
            A = by_video[va]
            for j, vb in enumerate(videos):
                if i == j:
                    mat[i, j] = 1.0
                    continue
                B = by_video[vb]

                bests = []
                for a in A:
                    best = 0.0
                    for b in B:
                        score, _ = compute_pair_similarity(a, b, weights)
                        if score > best:
                            best = score
                    bests.append(best)

                mat[i, j] = float(np.mean(bests)) if bests else 0.0

        mat_df = pd.DataFrame(mat, index=videos, columns=videos)
        st.dataframe(mat_df.style.format("{:.3f}"), use_container_width=True)

# ---------------------------------
# Metric definitions
# ---------------------------------
with st.expander("Metric definitions (what the app is computing)"):
    st.markdown(
        """
- **Structural alignment (SSIM):** similarity of grayscale structure after resizing.
- **Color histogram:** correlation between normalized per-channel color histograms.
- **Entropy similarity:** compares Shannon entropy of grayscale; closer entropy → more similar.
- **Edge complexity:** compares edge density from Canny edges.
- **Texture correlation:** compares LBP texture histograms (cosine similarity).
- **Brightness similarity:** compares mean grayscale brightness.
- **Hue distribution:** correlation between hue histograms in HSV space.
        """
    )
