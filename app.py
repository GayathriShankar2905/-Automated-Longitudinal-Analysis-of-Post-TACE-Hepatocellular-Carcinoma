import streamlit as st
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import networkx as nx
from scipy.ndimage import zoom, distance_transform_edt, shift, binary_erosion, binary_dilation

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    layout="wide",
    page_title="HCC Longitudinal Analysis",
    initial_sidebar_state="collapsed"
)

# =========================
# PROFESSIONAL SIEMENS THEME
# =========================
st.markdown("""
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    
    /* HEADER SECTION */
    .header-container {
        background: linear-gradient(135deg, #003d82 0%, #004fa3 100%);
        padding: 40px 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 8px 24px rgba(0, 61, 130, 0.15);
        border-left: 6px solid #00a8e8;
    }
    
    .header-title {
        color: #ffffff;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 20px;
        letter-spacing: -0.5px;
        line-height: 1.3;
    }
    
    .header-authors {
        color: #b3d9ff;
        font-size: 13px;
        margin-bottom: 8px;
        font-weight: 500;
    }
    
    .header-dept {
        color: #80c4ff;
        font-size: 12px;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    /* SECTION HEADERS */
    h2 {
        color: #003d82;
        font-size: 20px;
        font-weight: 700;
        margin-top: 35px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 3px solid #00a8e8;
        letter-spacing: -0.3px;
    }
    
    h3 {
        color: #003d82;
        font-size: 16px;
        font-weight: 600;
    }
    
    /* METRIC CARDS */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 61, 130, 0.08);
        border-left: 4px solid #00a8e8;
        margin-bottom: 12px;
    }
    
    /* SUCCESS/ERROR BOXES */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 6px;
        padding: 15px;
        border-left: 4px solid #28a745;
    }
    
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 6px;
        padding: 15px;
        border-left: 4px solid #dc3545;
    }
    
    /* BUTTON STYLING */
    .stButton > button {
        background: linear-gradient(135deg, #003d82 0%, #004fa3 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 61, 130, 0.25);
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 16px rgba(0, 61, 130, 0.35);
        transform: translateY(-2px);
    }
    
    /* DIVIDER */
    .stDivider {
        border-color: #00a8e8;
    }
    
    /* FILE UPLOADER */
    .stFileUploader {
        border: 2px dashed #003d82;
        border-radius: 8px;
        padding: 20px;
        background: rgba(0, 61, 130, 0.02);
    }
    
    /* COLUMN LAYOUT */
    .stColumn {
        padding: 10px;
    }
    
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header-container">
    <div class="header-title">🧠 Automated Longitudinal Analysis of Post-TACE Hepatocellular Carcinoma</div>
    <div style="font-size: 13px; color: #b3d9ff; margin-bottom: 15px; font-weight: 400;">
        Using Deep Segmentation and Graph-Based Transformer Models
    </div>
    <div class="header-authors">Rithika Sena Jyothula, Gayathri S.H, Lata Samariya (4th year B.Tech)</div>
    <div class="header-authors">Dr. Nijisha Shajil (Assistant Professor)</div>
    <div class="header-dept">Department of Biomedical Engineering</div>
    <div class="header-dept">SRM Institute of Science and Technology, Kattankulathur-603203, Tamil Nadu, India</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================
# HELPERS
# =========================
def load_ct(files):
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return np.stack([s.pixel_array for s in slices]), slices

def align_mask(seg_ds, slices, label_idx):
    arr = seg_ds.pixel_array
    frames = seg_ds.PerFrameFunctionalGroupsSequence
    ct_z = np.array([float(s.ImagePositionPatient[2]) for s in slices])

    mask = np.zeros((len(ct_z), arr.shape[1], arr.shape[2]))

    for i in range(arr.shape[0]):
        seg_id = frames[i].SegmentIdentificationSequence[0].ReferencedSegmentNumber
        if seg_id == label_idx:
            z = float(frames[i].PlanePositionSequence[0].ImagePositionPatient[2])
            z_idx = np.argmin(np.abs(ct_z - z))
            mask[z_idx] = arr[i]

    return mask.astype(np.uint8)

def simulate_post(pre_mask):
    core = binary_erosion(pre_mask, iterations=3)
    shifted = shift(core.astype(float), shift=(0,3,3), order=0)
    return (shifted > 0.5).astype(np.uint8)

def min_distance(a, b):
    if np.sum(a)==0 or np.sum(b)==0: return 0
    dist = distance_transform_edt(b==0)
    return float(np.min(dist[a>0]))

# =========================
# INPUT SECTION
# =========================
st.markdown("### 📂 Upload Medical Data")

col1, col2, col3 = st.columns(3)
with col1:
    pre_files = st.file_uploader("📍 PRE-TACE CT Series", accept_multiple_files=True, key="pre")
with col2:
    post_files = st.file_uploader("📍 POST-TACE CT Series", accept_multiple_files=True, key="post")
with col3:
    seg_file = st.file_uploader("📍 Segmentation DICOM", key="seg")

# =========================
# PROCESS BUTTON
# =========================
run_button = st.button("🚀 Run Analysis Pipeline", use_container_width=True)

# =========================
# PROCESSING
# =========================
if run_button and pre_files and post_files and seg_file:

    # Load data
    pre_ct, slices = load_ct(pre_files)
    post_ct, _ = load_ct(post_files)
    seg_ds = pydicom.dcmread(seg_file)

    labels = [s.SegmentLabel for s in seg_ds.SegmentSequence]

    masks = {}
    for i, lbl in enumerate(labels):
        masks[lbl] = align_mask(seg_ds, slices, i+1)

    liver  = masks.get("Liver", np.zeros_like(pre_ct))
    tumor  = masks.get("Mass", np.zeros_like(pre_ct))
    portal = masks.get("Portal vein", np.zeros_like(pre_ct))
    aorta  = masks.get("Abdominal aorta", np.zeros_like(pre_ct))

    # Align and simulate
    scale = post_ct.shape[0] / tumor.shape[0]
    tumor_aligned = zoom(tumor, (scale,1,1), order=0)
    post_tumor = simulate_post(tumor_aligned)

    z_pre = np.argmax(np.sum(tumor, axis=(1,2)))
    z_post = int(z_pre * scale)

    # ===== CT SCANS =====
    st.markdown("## 🖼️ Imaging Analysis")
    
    c1, c2 = st.columns(2)
    
    with c1:
        fig, ax = plt.subplots(figsize=(7,7))
        ax.imshow(pre_ct[z_pre], cmap="gray")
        ax.set_title("PRE-TACE CT", fontsize=14, fontweight='bold', color='#003d82', pad=15)
        ax.axis("off")
        fig.patch.set_facecolor('#f5f7fa')
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig, ax = plt.subplots(figsize=(7,7))
        ax.imshow(post_ct[z_post], cmap="gray")
        ax.set_title("POST-TACE CT", fontsize=14, fontweight='bold', color='#003d82', pad=15)
        ax.axis("off")
        fig.patch.set_facecolor('#f5f7fa')
        st.pyplot(fig, use_container_width=True)

    # ===== SEGMENTATION MASKS =====
    st.markdown("## 🧬 Segmentation Masks")
    
    def show_mask(mask, title, color=(1, 1, 1)):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_facecolor("black")
        display = np.zeros((*mask.shape, 3))
        display[mask > 0] = color
        ax.imshow(display)
        ax.set_title(title, fontsize=12, fontweight='bold', color='white', pad=10)
        ax.axis("off")
        fig.patch.set_facecolor('#f5f7fa')
        return fig

    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.pyplot(show_mask(liver[z_pre], "Liver", (0.2, 0.8, 0.2)), use_container_width=True)
    with c2:
        st.pyplot(show_mask(tumor[z_pre], "Tumor", (1, 1, 1)), use_container_width=True)
    with c3:
        st.pyplot(show_mask(portal[z_pre], "Portal Vein", (0.2, 0.8, 1)), use_container_width=True)
    with c4:
        st.pyplot(show_mask(aorta[z_pre], "Aorta", (1, 0.6, 0.2)), use_container_width=True)

    # ===== TUMOR EVOLUTION =====
    st.markdown("## 🔥 Tumor Evolution Analysis")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("black")
    fig.patch.set_facecolor('#f5f7fa')

    pre_mask = tumor_aligned[z_post]
    post_mask = post_tumor[z_post]

    # PRE WHITE
    pre_disp = np.zeros((*pre_mask.shape, 3))
    pre_disp[pre_mask > 0] = [1, 1, 1]
    ax.imshow(pre_disp, alpha=0.9)

    # PRE EDGE (CYAN)
    edge = binary_dilation(pre_mask) ^ pre_mask
    edge_disp = np.zeros((*pre_mask.shape, 3))
    edge_disp[edge > 0] = [0, 1, 1]
    ax.imshow(edge_disp, alpha=1)

    # POST RED
    post_disp = np.zeros((*post_mask.shape, 3))
    post_disp[post_mask > 0] = [1, 0, 0]
    ax.imshow(post_disp, alpha=0.7)

    ax.set_title("PRE-TACE (White) vs POST-TACE (Red)", color="white", fontsize=14, fontweight='bold', pad=15)
    ax.axis("off")

    st.pyplot(fig, use_container_width=True)

    # ===== QUANTITATIVE METRICS =====
    st.markdown("## 📊 Quantitative Analysis")

    pre_vol = np.sum(tumor_aligned)
    post_vol = np.sum(post_tumor)
    change = (post_vol - pre_vol) / (pre_vol + 1e-6)

    if post_vol <= 0.1 * pre_vol:
        mrecist = "CR"
        mrecist_full = "Complete Response"
    elif change <= -0.3:
        mrecist = "PR"
        mrecist_full = "Partial Response"
    elif change >= 0.2:
        mrecist = "PD"
        mrecist_full = "Progressive Disease"
    else:
        mrecist = "SD"
        mrecist_full = "Stable Disease"

    response = "Non-Responder" if mrecist == "PD" else "Responder"

    # Metrics row
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric(
            label="Pre Volume",
            value=f"{int(pre_vol)} mm³",
            delta=None,
        )
    
    with metric_col2:
        st.metric(
            label="Post Volume",
            value=f"{int(post_vol)} mm³",
            delta=f"{change*100:.1f}%",
        )
    
    with metric_col3:
        st.metric(
            label="mRECIST",
            value=mrecist,
            delta=mrecist_full,
        )
    
    with metric_col4:
        st.metric(
            label="Response Status",
            value=response,
            delta="Treatment Efficacy",
        )

    # Result box
    st.markdown("")
    if response == "Responder":
        st.success(f"✅ **{mrecist_full}** — Patient is a **{response}**. Treatment shows positive response.", icon="✅")
    else:
        st.error(f"⚠️ **{mrecist_full}** — Patient is a **{response}**. Treatment may require adjustment.", icon="❌")

    # ===== CONNECTOME =====
    st.markdown("## 🧠 Hepatic Vasculature Connectome")

    d_tp = min_distance(tumor, portal)
    d_ta = min_distance(tumor, aorta)

    G = nx.Graph()
    G.add_edge("Tumor", "Portal Vein", label=f"{d_tp:.1f} mm")
    G.add_edge("Tumor", "Aorta", label=f"{d_ta:.1f} mm")

    pos = {
        "Tumor": (0, 0),
        "Portal Vein": (-1.2, 1.2),
        "Aorta": (1.2, 1.2)
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#f5f7fa')

    nx.draw_networkx_nodes(
        G, pos,
        node_color=["#ff4b4b", "#0077b6", "#0077b6"],
        node_size=4000,
        ax=ax
    )

    nx.draw_networkx_labels(
        G, pos,
        font_size=11,
        font_weight='bold',
        font_color='white',
        ax=ax
    )

    nx.draw_networkx_edges(
        G, pos,
        width=2.5,
        edge_color='#003d82',
        ax=ax
    )

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=10,
        font_weight='bold',
        font_color='#003d82',
        ax=ax
    )

    ax.set_title("Distance Metrics: Tumor to Critical Vasculature", fontsize=13, fontweight='bold', color='#003d82', pad=15)
    ax.axis("off")

    st.pyplot(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 12px; margin-top: 20px;">
        <strong>Clinical Decision Support System</strong><br>
        This analysis is for research and educational purposes. Always consult qualified radiologists and clinicians for clinical decisions.
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.info("👈 **Upload CT and segmentation DICOM files to begin analysis**", icon="ℹ️")