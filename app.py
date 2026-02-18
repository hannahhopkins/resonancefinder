# app.py
# Frame Similarity Analyzer (UPLOAD ONLY)
# - Supports 1 video (single-video scope) or many (cross-video optional)
# - Frame labels: Visible "Video A · 00:12" + hover-only metadata (Option 3)
# - Similarity metrics: SSIM, color hist, entropy, edge, texture (LBP), brightness, hue
# - Optional: CLIP similarity (semantic/compositional resonance) via open_clip_torch
# - Optional: Scene clustering across all frames, capped at 20 clusters total

import os
import html
import tempfile
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern

# Optional: clustering
try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# Optional: CLIP
try:
    import torch
    import open_clip
    CLIP_OK = True
except Exception:
    CLIP_OK = False


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Frame Similarity (Uploads)", layout="wide")

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

def hover_label(text: str, tooltip: str, size_px: int = 13) -> None:
    """
    Renders a visible label with hover-only metadata (HTML title="...").
    """
    text_e = html.escape(text)
    tip_e = html.escape(tooltip)
    st.markdown(
        f"""
        <div title="{tip_e}"
             style="
                font-size:{size_px}px;
                margin-top:2px;
                white-space:nowrap;
                overflow:hidden;
                text-overflow:ellipsis;
                cursor:default;
                opacity:0.92;
             ">
          {text_e}
        </div>
        """,
        unsafe_allow_html=True,
    )

def build_hover_tooltip(
    r,
    similarity_score: Optional[float] = None,
    cluster_id: Optional[int] = None,
    clip_sim: Optional[float] = None,
) -> str:
    parts = [
        f"Frame ID: {r.frame_id}",
        f"Exact time: {r.t_sec:.6f} seconds",
        f"Frame index: {r.frame_index}",
        f"Entropy: {r.entropy:.6f}",
        f"Edge density: {r.edge:.6f}",
        f"Brightness: {r.brightness:.2f}",
    ]
    if similarity_score is not None:
        parts.append(f"Similarity score: {float(similarity_score):.6f}")
    if clip_sim is not None:
        parts.append(f"CLIP similarity: {float(clip_sim):.6f}")
    if cluster_id is not None:
        parts.append(f"Cluster: {int(cluster_id)}")
    parts.append(f"Source: {r.video_name}")
    return "\n".join(parts)

def render_frame_label(
    r,
    similarity_score: Optional[float] = None,
    cluster_id: Optional[int] = None,
    clip_sim: Optional[float] = None,
) -> None:
    visible = f"{video_letter_name(r.video_index)} · {format_time(r.t_sec)}"
    tooltip = build_hover_tooltip(r, similarity_score=similarity_score, cluster_id=cluster_id, clip_sim=clip_sim)
    hover_label(visible, tooltip, size_px=13)


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
# Frame record
# ----------------------------
@dataclass
class FrameRecord:
    frame_id: str
    video_name: str
    video_index: int
    t_sec: float
    frame_index: int
    thumb_path: str

    gray_small: np.ndarray
    color_hist: np.ndarray
    hue_hist: np.ndarray
    entropy: float
    edge: float
    texture_hist: np.ndarray
    brightness: float

    clip_emb: Optional[np.ndarray] = None


# ----------------------------
# Similarity
# ----------------------------
def compute_pair_similarity(
    A: FrameRecord,
    B: FrameRecord,
    weights: Dict[str, float],
    use_clip: bool,
) -> Tuple[float, Dict[str, float]]:
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

    comps: Dict[str, float] = {
        "structural_alignment": ssim_sim,
        "color_histogram": color_sim,
        "entropy_similarity": ent_sim,
        "edge_complexity": edge_sim,
        "texture_correlation": tex_sim,
        "brightness_similarity": bri_sim,
        "hue_distribution": hue_sim,
    }

    if use_clip and (A.clip_emb is not None) and (B.clip_emb is not None):
        c = float(np.dot(A.clip_emb, B.clip_emb))  # normalized embeddings
        c = max(-1.0, min(1.0, c))
        comps["clip_similarity"] = (c + 1.0) / 2.0

    wsum = sum(max(0.0, float(weights.get(k, 0.0))) for k in comps.keys()) + 1e-12
    score = sum(comps[k] * max(0.0, float(weights.get(k, 0.0))) for k in comps.keys()) / wsum
    return float(score), comps


# ----------------------------
# Upload handling + extraction caching
# ----------------------------
def _sha1_bytes(bts: bytes) -> str:
    return hashlib.sha1(bts).hexdigest()

def ensure_upload_tempfile(upload) -> str:
    """
    Persist upload to a temp file path that survives reruns via session_state.
    """
    bts = upload.getbuffer().tobytes()
    key = f"{upload.name}::{_sha1_bytes(bts)}"

    if "upload_paths" not in st.session_state:
        st.session_state.upload_paths = {}

    if key in st.session_state.upload_paths and os.path.exists(st.session_state.upload_paths[key]):
        return st.session_state.upload_paths[key]

    suffix = os.path.splitext(upload.name)[1] or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="upload_")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(bts)
    st.session_state.upload_paths[key] = path
    return path

def build_cache_key(
    sources: List[Tuple[str, str]],
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
    for path, name in sources:
        h.update(name.encode("utf-8"))
        try:
            stt = os.stat(path)
            h.update(str(stt.st_size).encode("utf-8"))
            h.update(str(int(stt.st_mtime)).encode("utf-8"))
        except Exception:
            h.update(path.encode("utf-8"))
    return h.hexdigest()


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

            # Thumbnail
            thumb = _resize_keep_aspect(frame, thumb_max_side)
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            thumb_img = Image.fromarray(thumb_rgb)

            frame_id = f"v{video_index}_f{frame_idx}"
            thumb_path = os.path.join(thumbs_dir, f"{frame_id}.jpg")
            thumb_img.save(thumb_path, quality=90)

            # Analysis frame
            analysis_bgr = _resize_keep_aspect(frame, analysis_max_side)
            gray = _to_gray(analysis_bgr).astype(np.uint8)

            frames.append(
                FrameRecord(
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
                    clip_emb=None,
                )
            )

            grabbed += 1
            if grabbed >= max_frames:
                break

        frame_idx += 1
        if total_frames and frame_idx >= total_frames:
            break

    cap.release()
    return frames


# ----------------------------
# CLIP
# ----------------------------
def _clip_device() -> str:
    if not CLIP_OK:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@st.cache_resource(show_spinner=False)
def load_clip_model(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
    if not CLIP_OK:
        return None, None, None
    device = _clip_device()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    return model, preprocess, device

def compute_clip_embeddings_for_records(records: List[FrameRecord]) -> None:
    """
    Computes CLIP embeddings from thumbnails (fast) and caches by frame_id in session_state.
    """
    model, preprocess, device = load_clip_model()
    if model is None:
        return

    if "clip_emb_cache" not in st.session_state:
        st.session_state.clip_emb_cache = {}

    batch = []
    batch_recs: List[FrameRecord] = []

    def flush():
        if not batch:
            return
        imgs = torch.stack(batch).to(device)
        with torch.no_grad():
            feats = model.encode_image(imgs)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
        feats = feats.detach().float().cpu().numpy().astype(np.float32)
        for rec, emb in zip(batch_recs, feats):
            st.session_state.clip_emb_cache[rec.frame_id] = emb
            rec.clip_emb = emb
        batch.clear()
        batch_recs.clear()

    bs = 32 if device in ("cuda", "mps") else 16

    for r in records:
        if r.frame_id in st.session_state.clip_emb_cache:
            r.clip_emb = st.session_state.clip_emb_cache[r.frame_id]
            continue
        try:
            pil = Image.open(r.thumb_path).convert("RGB")
            img_t = preprocess(pil)
            batch.append(img_t)
            batch_recs.append(r)
            if len(batch) >= bs:
                flush()
        except Exception:
            r.clip_emb = None

    flush()


# ----------------------------
# Clustering (capped at 20 clusters total)
# ----------------------------
def build_feature_matrix_for_clustering(records: List[FrameRecord]) -> np.ndarray:
    feats = []
    for r in records:
        vec = np.concatenate(
            [
                r.color_hist,
                r.hue_hist,
                r.texture_hist,
                np.array([r.entropy / 8.0, r.edge, r.brightness / 255.0], dtype=np.float32),
            ]
        )
        feats.append(vec)
    X = np.vstack(feats).astype(np.float32)
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def cluster_frames_capped(records: List[FrameRecord], distance_threshold: float, max_clusters: int = 20) -> np.ndarray:
    """
    Agglomerative clustering with an automatic cap on the number of clusters.
    If the requested distance_threshold yields too many clusters, we increase
    the threshold until clusters <= max_clusters, then binary-search the smallest
    threshold that satisfies the cap.
    """
    X = build_feature_matrix_for_clustering(records)

    def run(th: float) -> np.ndarray:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(th),
            linkage="ward",
        )
        return clustering.fit_predict(X)

    labels = run(distance_threshold)
    n = len(set(map(int, labels.tolist() if hasattr(labels, "tolist") else labels)))
    if n <= max_clusters:
        return labels

    lo = float(distance_threshold)
    hi = float(distance_threshold)

    # Expand upper bound until we satisfy cap
    labels_hi = labels
    for _ in range(12):
        hi *= 1.5
        labels_hi = run(hi)
        n_hi = len(set(map(int, labels_hi.tolist() if hasattr(labels_hi, "tolist") else labels_hi)))
        if n_hi <= max_clusters:
            break
    else:
        # Couldn't get under cap; return best effort
        return labels_hi

    # Binary search smallest threshold that satisfies cap
    best = labels_hi
    for _ in range(20):
        mid = (lo + hi) / 2.0
        labels_mid = run(mid)
        n_mid = len(set(map(int, labels_mid.tolist() if hasattr(labels_mid, "tolist") else labels_mid)))
        if n_mid <= max_clusters:
            best = labels_mid
            hi = mid
        else:
            lo = mid

    return best


# ----------------------------
# UI
# ----------------------------
st.title("Frame Similarity Analyzer (Uploads)")

with st.sidebar:
    st.header("1) Upload videos")
    uploads = st.file_uploader(
        "Upload video file(s). One video is fine; multiple enables cross-video comparisons.",
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

    st.header("4) Similarity metrics")
    st.caption("Weights control each metric’s contribution to the final score.")
    weights: Dict[str, float] = {
        "structural_alignment": st.slider("Structural alignment (SSIM)", 0.0, 3.0, 0.8, 0.1),
        "color_histogram": st.slider("Color histogram", 0.0, 3.0, 1.2, 0.1),
        "entropy_similarity": st.slider("Entropy similarity", 0.0, 3.0, 0.6, 0.1),
        "edge_complexity": st.slider("Edge complexity", 0.0, 3.0, 0.9, 0.1),
        "texture_correlation": st.slider("Texture correlation (LBP)", 0.0, 3.0, 1.1, 0.1),
        "brightness_similarity": st.slider("Brightness similarity", 0.0, 3.0, 0.8, 0.1),
        "hue_distribution": st.slider("Hue distribution", 0.0, 3.0, 1.2, 0.1),
    }

    st.subheader("CLIP (semantic resonance)")
    if not CLIP_OK:
        st.info("CLIP disabled (missing `torch` and/or `open_clip_torch`).")
        enable_clip = False
    else:
        enable_clip = st.checkbox("Enable CLIP similarity", value=True)
        if enable_clip:
            weights["clip_similarity"] = st.slider("CLIP similarity weight", 0.0, 6.0, 3.0, 0.1)

    st.header("5) Retrieval")
    top_k = st.slider("Top-K matches", min_value=3, max_value=50, value=12, step=1)

if not uploads:
    st.info("Upload at least one video to begin.")
    st.stop()

# Build sources list (path, display_name)
sources: List[Tuple[str, str]] = []
for u in uploads:
    p = ensure_upload_tempfile(u)
    sources.append((p, u.name))

multi_video = len(sources) > 1

# Extraction cache key
cache_key = build_cache_key(
    sources=sources,
    mode=mode,
    interval_value=float(interval_value),
    max_frames=int(max_frames),
    thumb_max_side=int(thumb_max_side),
    analysis_max_side=int(analysis_max_side),
)

# Extract (cached in session)
if "cache_key" not in st.session_state or st.session_state.cache_key != cache_key:
    thumbs_dir = tempfile.mkdtemp(prefix="thumbs_")
    records: List[FrameRecord] = []
    with st.spinner("Extracting frames and computing descriptors..."):
        for i, (path, name) in enumerate(sources):
            recs = extract_frames(
                video_path=path,
                video_name=name,
                video_index=i,
                mode=mode,
                interval_value=float(interval_value),
                max_frames=int(max_frames),
                thumb_max_side=int(thumb_max_side),
                analysis_max_side=int(analysis_max_side),
                thumbs_dir=thumbs_dir,
            )
            records.extend(recs)

    st.session_state.cache_key = cache_key
    st.session_state.records = records
    st.session_state.thumbs_dir = thumbs_dir
    st.session_state.cluster_labels = None
else:
    records = st.session_state.records
    thumbs_dir = st.session_state.thumbs_dir

if not records:
    st.error("No frames were extracted. Try a smaller interval or a higher max-frames-per-video.")
    st.stop()

# Optional: CLIP embeddings
if enable_clip and CLIP_OK:
    with st.spinner("Computing CLIP embeddings for extracted frames..."):
        compute_clip_embeddings_for_records(records)

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

# cluster map
cluster_map: Optional[Dict[str, int]] = None
if st.session_state.get("cluster_labels") is not None:
    labels = st.session_state.cluster_labels
    cluster_map = {records[i].frame_id: int(labels[i]) for i in range(len(records))}


# ---------------------------------
# Extracted frames browser
# ---------------------------------
st.subheader("Extracted frames")

c1, c2, c3 = st.columns([1.25, 1.0, 2.75])

with c1:
    video_filter = st.multiselect(
        "Filter by source video",
        options=sorted(df["video"].unique().tolist()),
        default=sorted(df["video"].unique().tolist()),
    )

with c2:
    df_f = df[df["video"].isin(video_filter)].copy()
    df_f = df_f.sort_values(["video_index", "time_sec", "frame_index"]).reset_index(drop=True)
    st.metric("Frames (filtered)", len(df_f))

with c3:
    if multi_video:
        st.caption("Multiple videos loaded. Similarity search can span all videos or stay within one.")
    else:
        st.caption("Single-video mode: similarity search runs within the one uploaded video.")

max_grid = 60
grid_df = df_f.head(max_grid)

cols = st.columns(6)
for i, row in enumerate(grid_df.itertuples(index=False)):
    with cols[i % 6]:
        st.image(row.thumb_path, use_container_width=True)
        r = rec_by_id[row.frame_id]
        cid = cluster_map.get(r.frame_id) if cluster_map else None
        render_frame_label(r, cluster_id=cid)

st.divider()

# ---------------------------------
# Similarity search
# ---------------------------------
st.subheader("Similarity search")

left, right = st.columns([1.0, 2.0], vertical_alignment="top")

with left:
    if multi_video:
        scope = st.radio("Search scope", ["All videos", "This video only"], index=0, horizontal=True)
    else:
        scope = "This video only"
        st.caption("Scope: This video only")

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
    if scope == "All videos":
        exclude_same_video = st.checkbox("Exclude frames from the same video", value=False)
    else:
        exclude_same_video = False

with right:
    sims = []
    for r in records:
        if exclude_same_frame and r.frame_id == query.frame_id:
            continue
        if scope == "This video only" and r.video_index != query.video_index:
            continue
        if scope == "All videos" and exclude_same_video and r.video_index == query.video_index:
            continue

        score, comps = compute_pair_similarity(query, r, weights, use_clip=bool(enable_clip and CLIP_OK))
        sims.append(
            {
                "score": score,
                "frame_id": r.frame_id,
                "video": r.video_name,
                "time_sec": r.t_sec,
                "frame_index": r.frame_index,
                "clip_sim": comps.get("clip_similarity", None),
                **{f"m_{k}": v for k, v in comps.items()},
                "thumb_path": r.thumb_path,
            }
        )

    sims_df = pd.DataFrame(sims).sort_values("score", ascending=False).head(int(top_k)).reset_index(drop=True)

    if sims_df.empty:
        st.warning("No matches found under the current settings.")
    else:
        st.write("Top matches")
        gallery_cols = st.columns(4)
        for i, row in enumerate(sims_df.itertuples(index=False)):
            with gallery_cols[i % 4]:
                st.image(row.thumb_path, use_container_width=True)
                rr = rec_by_id[row.frame_id]
                cid2 = cluster_map.get(rr.frame_id) if cluster_map else None
                render_frame_label(
                    rr,
                    similarity_score=row.score,
                    cluster_id=cid2,
                    clip_sim=row.clip_sim if (enable_clip and CLIP_OK) else None,
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
            if enable_clip and CLIP_OK:
                show_cols.append("m_clip_similarity")
            st.dataframe(sims_df[[c for c in show_cols if c in sims_df.columns]], use_container_width=True)

st.divider()

# ---------------------------------
# Scene clustering (optional, capped at 20 clusters)
# ---------------------------------
st.subheader("Scene clustering (optional)")

if not SKLEARN_OK:
    st.info("Clustering disabled (missing scikit-learn). Add `scikit-learn` to requirements.txt to enable.")
else:
    st.caption("Clusters are capped at 20 total; if your threshold yields more, the app will auto-relax it.")
    cluster_threshold = st.slider(
        "Cluster sensitivity (lower = more clusters, higher = fewer clusters)",
        min_value=1.0,
        max_value=15.0,
        value=5.0,
        step=0.5,
    )
    do_cluster = st.button("Cluster frames (cap = 20)")

    if do_cluster:
        with st.spinner("Clustering frames..."):
            labels = cluster_frames_capped(records, distance_threshold=float(cluster_threshold), max_clusters=20)
            st.session_state.cluster_labels = labels
            cluster_map = {records[i].frame_id: int(labels[i]) for i in range(len(records))}

        n_clusters = len(set(cluster_map.values()))
        st.success(f"Detected {n_clusters} clusters (capped at 20). Hover labels now show cluster numbers.")

        st.caption("Cluster preview (first ~18 frames per cluster)")
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
# Cross-video similarity matrix (optional)
# ---------------------------------
st.subheader("Cross-video similarity matrix (optional)")

if not multi_video:
    st.info("Upload 2+ videos to enable the cross-video similarity matrix.")
else:
    st.caption("For each frame in Video A, take its best match in Video B, then average.")
    do_matrix = st.checkbox("Compute cross-video similarity matrix", value=False)
    if do_matrix:
        with st.spinner("Computing matrix... (can be slow)"):
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
                            score, _ = compute_pair_similarity(a, b, weights, use_clip=bool(enable_clip and CLIP_OK))
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
- **Structural alignment (SSIM):** grayscale structural similarity (best for near-duplicates).
- **Color histogram:** palette similarity via RGB distributions.
- **Hue distribution:** palette similarity via HSV hue distributions.
- **Entropy similarity:** global image complexity similarity.
- **Edge complexity:** edge-density similarity (structural complexity).
- **Texture correlation (LBP):** texture/material similarity.
- **Brightness similarity:** tonal similarity via mean luminance.
- **CLIP similarity (optional):** semantic + compositional resonance (best for “visual resonance” across different footage).
        """
    )
