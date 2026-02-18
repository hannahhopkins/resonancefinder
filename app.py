# app.py
import io
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Multi-Video Frame Similarity", layout="wide")

LBP_P = 16
LBP_R = 2
LBP_METHOD = "uniform"

# ----------------------------
# Helpers: image + descriptors
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

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def shannon_entropy_gray(gray: np.ndarray) -> float:
    # grayscale histogram entropy (base-2) in [0, 8] roughly for 8-bit
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
    # normalized 3-channel hist flattened
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
    # For uniform LBP, number of patterns is P+2
    n_bins = LBP_P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=False)
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-12)
    return hist

def corr_sim(a: np.ndarray, b: np.ndarray) -> float:
    # cosine similarity in [0,1] after mapping from [-1,1]
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    cos = float(np.dot(a, b) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    return (cos + 1.0) / 2.0

def hist_corr_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Pearson corr-ish using OpenCV compareHist correlation on 1D
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    # compareHist expects shape (N,1) or (1,N)
    a2 = a.reshape(-1, 1)
    b2 = b.reshape(-1, 1)
    c = float(cv2.compareHist(a2, b2, cv2.HISTCMP_CORREL))
    c = max(-1.0, min(1.0, c))
    return (c + 1.0) / 2.0

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

def compute_pair_similarity(
    A: FrameRecord,
    B: FrameRecord,
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    # Structural alignment: SSIM on small grayscale
    try:
        s = float(ssim(A.gray_small, B.gray_small, data_range=255))
        s = max(-1.0, min(1.0, s))
        ssim_sim = (s + 1.0) / 2.0  # map to [0,1]
    except Exception:
        ssim_sim = 0.0

    # Color histogram similarity
    color_sim = hist_corr_sim(A.color_hist, B.color_hist)

    # Entropy similarity
    # entropy roughly 0..8, similarity = 1 - normalized absolute difference
    e_diff = abs(A.entropy - B.entropy)
    ent_sim = 1.0 - min(1.0, e_diff / 8.0)

    # Edge complexity similarity (edge density 0..1)
    edge_sim = 1.0 - min(1.0, abs(A.edge - B.edge) / 1.0)

    # Texture correlation (LBP hist cosine sim)
    tex_sim = corr_sim(A.texture_hist, B.texture_hist)

    # Brightness similarity (0..255)
    bri_sim = 1.0 - min(1.0, abs(A.brightness - B.brightness) / 255.0)

    # Hue distribution similarity
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

    # weighted average
    wsum = sum(max(0.0, _safe_float(weights.get(k, 0.0))) for k in comps.keys()) + 1e-12
    score = sum(comps[k] * max(0.0, _safe_float(weights.get(k, 0.0))) for k in comps.keys()) / wsum
    return float(score), comps


def build_feature_matrix(records):
    """
    Converts FrameRecord descriptors into a single feature matrix for clustering.
    """

    features = []

    for r in records:
        vec = np.concatenate([
            r.color_hist,                 # color distribution
            r.hue_hist,                   # hue distribution
            r.texture_hist,               # texture
            np.array([
                r.entropy / 8.0,
                r.edge,
                r.brightness / 255.0,
            ], dtype=np.float32)
        ])

        features.append(vec)

    X = np.vstack(features)

    # normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def cluster_frames(records, distance_threshold=5.0):
    """
    Clusters frames into visual scenes across ALL videos.
    Uses hierarchical clustering so number of clusters is automatically determined.
    """

    X = build_feature_matrix(records)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="ward"
    )

    labels = clustering.fit_predict(X)

    return labels


def compute_cluster_representatives(records, labels):
    """
    Returns one representative frame per cluster (closest to centroid).
    """

    reps = {}

    X = build_feature_matrix(records)

    for cluster_id in np.unique(labels):

        indices = np.where(labels == cluster_id)[0]
        cluster_vectors = X[indices]

        centroid = cluster_vectors.mean(axis=0)

        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)

        best_index = indices[np.argmin(distances)]

        reps[cluster_id] = records[best_index]

    return reps


# ----------------------------
# Video -> frames
# ----------------------------
def write_uploaded_to_temp(uploaded_file) -> str:
    # store to temp file so cv2 can read it
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

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

    # If total_frames unknown, just iterate until failure.
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

            # analysis frame (smaller)
            analysis_bgr = _resize_keep_aspect(frame, analysis_max_side)
            gray = _to_gray(analysis_bgr)

            rec = FrameRecord(
                frame_id=frame_id,
                video_name=video_name,
                video_index=video_index,
                t_sec=float(t_sec),
                frame_index=int(frame_idx),
                thumb_path=thumb_path,
                gray_small=gray.astype(np.uint8),
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

        # guard for certain codecs returning nonsense CAP_PROP_FRAME_COUNT
        if total_frames and frame_idx >= total_frames:
            break

    cap.release()
    return frames

@st.cache_data(show_spinner=False)
def process_videos_cached(
    file_bytes_and_names: List[Tuple[bytes, str]],
    mode: str,
    interval_value: float,
    max_frames: int,
    thumb_max_side: int,
    analysis_max_side: int,
) -> Tuple[List[Dict], str]:
    """
    Returns:
      - list of serializable frame rows with descriptors saved separately in memory via session (we reconstruct records later)
      - thumbs_dir used
    """
    thumbs_dir = tempfile.mkdtemp(prefix="thumbs_")

    temp_paths = []
    try:
        all_records: List[FrameRecord] = []
        for i, (bts, name) in enumerate(file_bytes_and_names):
            # write bytes to temp path
            suffix = os.path.splitext(name)[1] or ".mp4"
            fd, path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            with open(path, "wb") as f:
                f.write(bts)
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

        # Make a minimal serializable representation; actual arrays will be stored in session_state separately.
        rows = []
        for r in all_records:
            rows.append(
                dict(
                    frame_id=r.frame_id,
                    video_name=r.video_name,
                    video_index=r.video_index,
                    t_sec=r.t_sec,
                    frame_index=r.frame_index,
                    thumb_path=r.thumb_path,
                    # descriptors: serialize scalars only
                    entropy=r.entropy,
                    edge=r.edge,
                    brightness=r.brightness,
                )
            )

        # NOTE: we can’t reliably cache numpy arrays in all Streamlit environments with great performance,
        # so we’ll reconstruct full FrameRecord objects in session_state right after this cache call.
        return rows, thumbs_dir
    finally:
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

def rebuild_records_from_rows(
    rows: List[Dict],
    thumbs_dir: str,
    file_bytes_and_names: List[Tuple[bytes, str]],
    mode: str,
    interval_value: float,
    max_frames: int,
    thumb_max_side: int,
    analysis_max_side: int,
) -> List[FrameRecord]:
    """
    Re-run extraction to rebuild numpy descriptors in-memory.
    (We keep caching for the expensive control flow & thumbs, but ensure descriptors are correct in-session.)
    """
    # We’ll extract again, but this time we can reuse thumbs_dir by pointing extraction there.
    # To avoid duplicating thumbnails, extraction will overwrite same names (same frame_id) harmlessly.
    temp_paths = []
    all_records: List[FrameRecord] = []
    try:
        for i, (bts, name) in enumerate(file_bytes_and_names):
            suffix = os.path.splitext(name)[1] or ".mp4"
            fd, path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            with open(path, "wb") as f:
                f.write(bts)
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
        return all_records
    finally:
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

st.divider()
st.subheader("Scene clustering across all videos")

cluster_threshold = st.slider(
    "Cluster sensitivity (lower = more clusters, higher = fewer clusters)",
    min_value=1.0,
    max_value=15.0,
    value=5.0,
    step=0.5,
)

if st.button("Cluster frames into scenes"):

    with st.spinner("Clustering frames..."):

        labels = cluster_frames(records, distance_threshold=cluster_threshold)

        representatives = compute_cluster_representatives(records, labels)

        n_clusters = len(representatives)

        st.success(f"{n_clusters} visual scenes detected")

        # display clusters
        for cluster_id in sorted(representatives.keys()):

            st.markdown(f"### Scene {cluster_id}")

            cluster_records = [
                r for i, r in enumerate(records)
                if labels[i] == cluster_id
            ]

            cols = st.columns(6)

            for i, r in enumerate(cluster_records[:18]):  # limit per cluster
                with cols[i % 6]:
                    st.image(r.thumb_path)
                    st.caption(f"{r.video_name} @ {r.t_sec:.2f}s")


# ----------------------------
# UI
# ----------------------------
st.title("Multi-Video Frame Similarity Analyzer")

with st.sidebar:
    st.header("1) Upload videos")
    uploads = st.file_uploader(
        "Upload 3 videos (or more). Common formats: mp4, mov, m4v.",
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

# Main
if not uploads:
    st.info("Upload your videos in the sidebar to begin.")
    st.stop()

if len(uploads) < 2:
    st.warning("Upload at least 2 videos to enable cross-video comparison.")
    st.stop()

# Turn uploads into stable bytes for caching
file_bytes_and_names = [(u.getbuffer().tobytes(), u.name) for u in uploads]

with st.spinner("Extracting frames and computing descriptors..."):
    rows, thumbs_dir = process_videos_cached(
        file_bytes_and_names=file_bytes_and_names,
        mode=mode,
        interval_value=float(interval_value),
        max_frames=int(max_frames),
        thumb_max_side=int(thumb_max_side),
        analysis_max_side=int(analysis_max_side),
    )
    # Rebuild full in-memory records for similarity computations
    records = rebuild_records_from_rows(
        rows=rows,
        thumbs_dir=thumbs_dir,
        file_bytes_and_names=file_bytes_and_names,
        mode=mode,
        interval_value=float(interval_value),
        max_frames=int(max_frames),
        thumb_max_side=int(thumb_max_side),
        analysis_max_side=int(analysis_max_side),
    )

if not records:
    st.error("No frames were extracted. Try a smaller interval or a higher max-frames-per-video.")
    st.stop()

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

# Build quick lookup
rec_by_id: Dict[str, FrameRecord] = {r.frame_id: r for r in records}

# ---------------------------------
# Section: Browse frames
# ---------------------------------
st.subheader("Extracted frames")
c1, c2, c3 = st.columns([1.2, 1.2, 2.6])

with c1:
    video_filter = st.multiselect(
        "Filter by video",
        options=sorted(df["video"].unique().tolist()),
        default=sorted(df["video"].unique().tolist()),
    )

with c2:
    df_f = df[df["video"].isin(video_filter)].copy()
    df_f = df_f.sort_values(["video_index", "time_sec", "frame_index"]).reset_index(drop=True)
    st.metric("Frames extracted (filtered)", len(df_f))

with c3:
    st.caption("Tip: pick a query frame below, then retrieve closest matches across *all* uploaded videos.")

# A grid of thumbnails (limited for UI sanity)
max_grid = 60
grid_df = df_f.head(max_grid)

cols = st.columns(6)
for i, row in enumerate(grid_df.itertuples(index=False)):
    col = cols[i % 6]
    with col:
        st.image(row.thumb_path, use_container_width=True)
        st.caption(f"{row.video}\n t={row.time_sec}s\n id={row.frame_id}")

st.divider()

# ---------------------------------
# Section: Query + retrieve
# ---------------------------------
st.subheader("Similarity search across all videos")

left, right = st.columns([1.0, 2.0], vertical_alignment="top")

with left:
    query_frame_id = st.selectbox(
        "Choose a query frame",
        options=df_f["frame_id"].tolist(),
        format_func=lambda fid: f"{fid} — {rec_by_id[fid].video_name} @ {rec_by_id[fid].t_sec:.2f}s",
    )
    query = rec_by_id[query_frame_id]
    st.image(query.thumb_path, use_container_width=True)
    st.caption(f"Query: {query.video_name} @ {query.t_sec:.2f}s (frame {query.frame_index})")

    exclude_same_frame = st.checkbox("Exclude the exact same frame_id from results", value=True)
    exclude_same_video = st.checkbox("Exclude frames from the same video", value=False)

with right:
    # Compute similarities
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
        # Show top matches as a gallery
        gallery_cols = st.columns(4)
        for i, row in enumerate(sims_df.itertuples(index=False)):
            with gallery_cols[i % 4]:
                st.image(row.thumb_path, use_container_width=True)
                st.caption(
                    f"Score: {row.score:.3f}\n"
                    f"{row.video}\n"
                    f"t={row.time_sec:.2f}s\n"
                    f"id={row.frame_id}"
                )

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
# Section: Cross-video summary
# ---------------------------------
st.subheader("Cross-video similarity summary (optional)")

st.caption(
    "This computes a rough 'how similar are these videos' score by taking, for each frame in Video A, "
    "the best match score found in Video B, then averaging those best-match scores."
)

do_summary = st.checkbox("Compute cross-video similarity matrix", value=False)

if do_summary:
    with st.spinner("Computing cross-video similarity matrix... (can be slow with many frames)"):
        videos = sorted(df["video"].unique().tolist())
        vid_to_idx = {v: i for i, v in enumerate(videos)}
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
                # Average of best matches A->B
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
# Notes
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

**Tip:** If your videos are long, start with a larger interval (e.g., every 2–5 seconds) and a max frames per video (e.g., 150–300),
then narrow down once you see where good matches cluster.
        """
    )
