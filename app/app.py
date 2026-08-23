import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from predict import (
    load_model,
    predict_emotion,
    MIN_AUDIO_SECONDS,
    CHUNK_DURATION_SECONDS,
    GLOBAL_MEAN,
    GLOBAL_STD,
    FIXED_LENGTH,
    USABLE_LENGTH,
    SAMPLE_RATE,
    NUM_CHUNKS,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sonic Compass — Music Emotion Recognition",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette tied to the valence-arousal circumplex itself
COLORS = {
    "bg": "#0E1016",
    "panel": "#171A24",
    "panel_alt": "#1D2130",
    "text": "#ECEDF4",
    "muted": "#9AA0B4",
    "accent": "#FF6B5B",
    "grid": "#2A2E3E",
    "happy": "#E4A11B",
    "tense": "#C0392B",
    "sad": "#4E7FD1",
    "calm": "#27AE60",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
.stApp {{
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
}}
header[data-testid="stHeader"] {{
    background-color: {COLORS['bg']};
}}
div[data-testid="stToolbar"] {{
    color: {COLORS['muted']};
}}
div[data-testid="stDecoration"] {{
    background-image: none;
    background-color: {COLORS['bg']};
}}
h1, h2, h3, .hero-title {{
    font-family: 'Space Grotesk', sans-serif !important;
}}
section[data-testid="stSidebar"] {{
    background-color: {COLORS['panel']};
    border-right: 1px solid {COLORS['grid']};
}}
.hero {{
    padding: 2.2rem 2rem 1.6rem 2rem;
    border-radius: 18px;
    background: linear-gradient(135deg, {COLORS['panel_alt']} 0%, {COLORS['panel']} 100%);
    border: 1px solid {COLORS['grid']};
    margin-bottom: 1.6rem;
}}
.hero-title {{
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    background: linear-gradient(90deg, {COLORS['happy']}, {COLORS['accent']});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.hero-sub {{
    color: {COLORS['muted']};
    font-size: 1.02rem;
    max-width: 640px;
}}
.card {{
    background-color: {COLORS['panel']};
    border: 1px solid {COLORS['grid']};
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
}}
.quadrant-badge {{
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.98rem;
    margin-bottom: 0.5rem;
}}
.emotion-card {{
    padding: 1.2rem 1.4rem;
    border-radius: 14px;
    background-color: {COLORS['panel']};
    border: 1px solid {COLORS['grid']};
    margin-top: 0.5rem;
}}
.small-muted, .small-muted * {{
    color: {COLORS['muted']} !important;
    font-size: 0.86rem;
}}
div[data-testid="stFileUploader"] {{
    background-color: {COLORS['panel']};
    border: 1px dashed {COLORS['grid']};
    border-radius: 14px;
    padding: 0.6rem;
}}
.stButton > button {{
    background-color: {COLORS['accent']};
    color: #10121A;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.6rem;
}}
.stButton > button:hover {{
    background-color: #ff8677;
    color: #10121A;
}}

/* ---- Contrast fixes: Streamlit's own widget text ignores .stApp's color
   and defaults to a dim grey that's hard to read on a dark background. ---- */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {COLORS['text']} !important;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {{
    color: #D5D8E5 !important;
}}
section[data-testid="stSidebar"] strong {{
    color: {COLORS['text']} !important;
}}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{
    color: {COLORS['muted']} !important;
}}
section[data-testid="stSidebar"] code {{
    color: #7EE787 !important;
    background-color: rgba(255,255,255,0.08) !important;
}}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{
    color: {COLORS['text']};
}}
[data-testid="stMetricLabel"] p {{
    color: #B7BBCB !important;
    font-weight: 500;
}}
[data-testid="stMetricValue"] {{
    color: {COLORS['text']} !important;
    font-weight: 700;
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {{
    color: {COLORS['text']} !important;
}}
[data-testid="stCaptionContainer"] {{
    color: {COLORS['muted']} !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span {{
    color: #3B3F4C !important;
}}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL COMPARISON (from the held-out test split in music-models.ipynb)
# ============================================================

METRICS_TABLE = pd.DataFrame([
    {"Model": "AST (Transformer)", "Valence MAE (1-9)": 0.7738, "Arousal MAE (1-9)": 0.7340,
     "Valence CCC": 0.5986, "Arousal CCC": 0.6756, "Valence R²": 0.3227, "Arousal R²": 0.4437},
    {"Model": "LSTM", "Valence MAE (1-9)": 0.7579, "Arousal MAE (1-9)": 0.7325,
     "Valence CCC": 0.5590, "Arousal CCC": 0.6389, "Valence R²": 0.3483, "Arousal R²": 0.4336},
    {"Model": "CNN", "Valence MAE (1-9)": 0.7959, "Arousal MAE (1-9)": 0.7915,
     "Valence CCC": 0.4814, "Arousal CCC": 0.5684, "Valence R²": 0.3049, "Arousal R²": 0.3739},
    {"Model": "Dummy (mean baseline)", "Valence MAE (1-9)": 0.9843, "Arousal MAE (1-9)": 1.0551,
     "Valence CCC": 0.0000, "Arousal CCC": 0.0000, "Valence R²": -0.0029, "Arousal R²": -0.0015},
    {"Model": "MLP", "Valence MAE (1-9)": 0.9863, "Arousal MAE (1-9)": 1.0546,
     "Valence CCC": 0.0000, "Arousal CCC": 0.0006, "Valence R²": -0.0130, "Arousal R²": -0.0016},
])


# ============================================================
# MODEL LOADING (delegates entirely to predict.py's verified logic)
# ============================================================

@st.cache_resource(show_spinner=False)
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with st.spinner("Downloading and loading the AST model — this can take a minute the first time…"):
        model = load_model(device)
    return model, device


# ============================================================
# SCALE + QUADRANT HELPERS
# ============================================================
#
# The model outputs raw valence/arousal in [0, 1]. DEAM's human annotations
# are on a 1-9 scale, and predict.py's own CLI output converts back with
# `raw * 8 + 1` — this app uses the same conversion so the displayed score
# means the same thing a DEAM annotator's rating would.

def to_deam_scale(raw_score: float) -> float:
    return raw_score * 8 + 1


QUADRANTS = {
    "happy": {
        "label": "Happy / Excited",
        "emoji": "😄",
        "color": COLORS["happy"],
        "description": (
            "High valence and high arousal. This is music that feels "
            "upbeat, joyful, or exhilarating — the kind that energizes a room."
        ),
        "tags": ["Upbeat", "Joyful", "Energetic", "Triumphant"],
    },
    "tense": {
        "label": "Tense / Angry",
        "emoji": "😠",
        "color": COLORS["tense"],
        "description": (
            "Low valence but high arousal. This is music with intensity "
            "and edge — driving, aggressive, or anxious rather than pleasant."
        ),
        "tags": ["Aggressive", "Anxious", "Intense", "Restless"],
    },
    "sad": {
        "label": "Sad / Depressed",
        "emoji": "😢",
        "color": COLORS["sad"],
        "description": (
            "Low valence and low arousal. This is music that feels heavy, "
            "melancholic, or subdued — slow and emotionally weighty."
        ),
        "tags": ["Melancholic", "Somber", "Weary", "Bleak"],
    },
    "calm": {
        "label": "Calm / Content",
        "emoji": "😌",
        "color": COLORS["calm"],
        "description": (
            "High valence but low arousal. This is music that feels "
            "peaceful, warm, or comforting — pleasant without urgency."
        ),
        "tags": ["Peaceful", "Relaxed", "Warm", "Serene"],
    },
}


def get_quadrant(valence_1_9: float, arousal_1_9: float) -> dict:
    mid = 5.0
    if valence_1_9 >= mid and arousal_1_9 >= mid:
        return QUADRANTS["happy"]
    if valence_1_9 < mid and arousal_1_9 >= mid:
        return QUADRANTS["tense"]
    if valence_1_9 < mid and arousal_1_9 < mid:
        return QUADRANTS["sad"]
    return QUADRANTS["calm"]


def intensity_label(valence_1_9: float, arousal_1_9: float) -> str:
    mid = 5.0
    distance = ((valence_1_9 - mid) ** 2 + (arousal_1_9 - mid) ** 2) ** 0.5
    max_distance = (4 ** 2 + 4 ** 2) ** 0.5
    ratio = distance / max_distance
    if ratio >= 0.66:
        return "strongly"
    if ratio >= 0.33:
        return "moderately"
    return "mildly"


# ============================================================
# PLOTS
# ============================================================

def dark_axis(title):
    return dict(
        title=title,
        range=[1, 9],
        zeroline=False,
        gridcolor=COLORS["grid"],
        color=COLORS["muted"],
    )


def build_circumplex_figure(valence, arousal, quadrant, trajectory=None):
    fig = go.Figure()

    quadrant_shapes = [
        (5, 5, 9, 9, COLORS["happy"]),
        (1, 5, 5, 9, COLORS["tense"]),
        (1, 1, 5, 5, COLORS["sad"]),
        (5, 1, 9, 5, COLORS["calm"]),
    ]
    for x0, y0, x1, y1, color in quadrant_shapes:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(width=0), fillcolor=color, opacity=0.10, layer="below")

    fig.add_shape(type="line", x0=1, y0=5, x1=9, y1=5,
                  line=dict(color=COLORS["grid"], width=1))
    fig.add_shape(type="line", x0=5, y0=1, x1=5, y1=9,
                  line=dict(color=COLORS["grid"], width=1))

    label_positions = {
        "Happy / Excited": (8.8, 8.7, "right", "top"),
        "Tense / Angry": (1.2, 8.7, "left", "top"),
        "Sad / Depressed": (1.2, 1.3, "left", "bottom"),
        "Calm / Content": (8.8, 1.3, "right", "bottom"),
    }
    for name, (x, y, xanchor, yanchor) in label_positions.items():
        fig.add_annotation(x=x, y=y, text=name, showarrow=False,
                           font=dict(size=11, color=COLORS["muted"]),
                           xanchor=xanchor, yanchor=yanchor)

    if trajectory is not None and len(trajectory) > 1:
        fig.add_trace(go.Scatter(
            x=trajectory[:, 0], y=trajectory[:, 1],
            mode="lines+markers",
            line=dict(color=COLORS["accent"], width=1.5),
            marker=dict(size=7, color=COLORS["accent"], opacity=0.55),
            opacity=0.5,
            name="Per-chunk",
            hovertemplate="Valence: %{x:.1f}<br>Arousal: %{y:.1f}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=[valence], y=[arousal], mode="markers",
        marker=dict(size=20, color=quadrant["color"], line=dict(width=2, color=COLORS["bg"])),
        name="Overall",
        hovertemplate=f"Valence: {valence:.1f}/9<br>Arousal: {arousal:.1f}/9<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dark_axis("Valence (negative → positive)"),
        yaxis=dark_axis("Arousal (calm → energetic)"),
        showlegend=False,
        height=440,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["muted"]),
    )
    return fig


def build_trajectory_figure(chunk_valence_1_9, chunk_arousal_1_9):
    times = [i * CHUNK_DURATION_SECONDS for i in range(len(chunk_valence_1_9))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=chunk_valence_1_9, mode="lines+markers", name="Valence",
        line=dict(color=COLORS["happy"], width=2.5), marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=times, y=chunk_arousal_1_9, mode="lines+markers", name="Arousal",
        line=dict(color=COLORS["sad"], width=2.5), marker=dict(size=6),
    ))
    fig.add_shape(type="line", x0=times[0], y0=5, x1=times[-1], y1=5,
                  line=dict(color=COLORS["grid"], width=1, dash="dash"))

    fig.update_layout(
        xaxis=dict(title="Time into clip (s)", gridcolor=COLORS["grid"], color=COLORS["muted"]),
        yaxis=dict(title="Score (1-9)", range=[1, 9], gridcolor=COLORS["grid"], color=COLORS["muted"]),
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["muted"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=COLORS["text"])),
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### About the model")
    st.markdown(
        "An **Audio Spectrogram Transformer** (`MIT/ast-finetuned-audioset-10-10-0.4593`), "
        "fine-tuned with a small regression head to predict **valence** and **arousal** "
        "on a 1–9 scale, trained on the **DEAM** dataset (1,802 songs, multiple listener "
        "annotations per song)."
    )

    st.markdown("### Why the transformer?")
    st.markdown(
        "Four architectures were trained and compared on the same held-out test set: "
        "a Transformer (AST), an LSTM, a CNN, and an MLP, against a mean-prediction "
        "baseline. The transformer produced the best **Concordance Correlation "
        "Coefficient (CCC)** on both valence and arousal — the metric that best "
        "captures whether predictions track the *true pattern* of a listener's "
        "emotional response, not just raw error."
    )
    with st.expander("Full test-set comparison"):
        st.dataframe(METRICS_TABLE.set_index("Model").style.format("{:.4f}"), use_container_width=True)
        st.caption("2,430 held-out chunk-level test examples. Lower MAE is better; higher CCC and R² are better.")

    with st.expander("How your audio is prepared"):
        st.markdown(f"""
- Loaded and resampled to **{SAMPLE_RATE:,} Hz**, mixed down to mono
- Converted to a **128-band mel spectrogram** in decibels
- Truncated to the first **{FIXED_LENGTH:,} frames**
- Normalized with the training set's mean/std ({GLOBAL_MEAN:.4f} / {GLOBAL_STD:.4f})
- Truncated to **{USABLE_LENGTH:,} frames** and split into **{NUM_CHUNKS} chunks** of
  1,024 frames (~{CHUNK_DURATION_SECONDS:.1f}s each)
- Each chunk is scored independently and averaged — identical to how
  training examples were built
- Clips shorter than **~{MIN_AUDIO_SECONDS:.0f} seconds** can't be processed,
  the same requirement the model was trained under
        """)

    st.markdown("---")
    st.caption("Model weights loaded from Hugging Face: rehan-hehe/ast-model")


# ============================================================
# HERO
# ============================================================

st.markdown(f"""
<div class="hero">
    <div class="hero-title">🎧 Sonic Compass</div>
    <div class="hero-sub">
        Upload a song and an Audio Spectrogram Transformer, fine-tuned on the DEAM dataset,
        will place it on the valence–arousal map of musical emotion — the same two-dimensional
        model of affect used in music psychology research.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# UPLOAD + ANALYZE
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an audio file (needs to be at least "
    f"~{MIN_AUDIO_SECONDS:.0f} seconds long)",
    type=["mp3", "wav", "flac", "m4a", "ogg"],
)

if uploaded_file is not None:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.audio(uploaded_file)
    with col_b:
        analyze = st.button("Analyze Emotion →", type="primary", use_container_width=True)

    if analyze:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        try:
            model, device = get_model()

            with st.spinner("Extracting mel spectrogram and running inference…"):
                valence_raw, arousal_raw, chunk_raw = predict_emotion(model, temp_path, device)

            valence = to_deam_scale(valence_raw)
            arousal = to_deam_scale(arousal_raw)
            chunk_1_9 = chunk_raw * 8 + 1  # [9, 2] on the same 1-9 scale

            quadrant = get_quadrant(valence, arousal)
            intensity = intensity_label(valence, arousal)

            st.success("Analysis complete!")

            res_col1, res_col2 = st.columns([1, 1.15])

            with res_col1:
                m1, m2 = st.columns(2)
                m1.metric("Valence", f"{valence:.2f} / 9")
                m2.metric("Arousal", f"{arousal:.2f} / 9")

                st.markdown(
                    f"""
                    <div class="emotion-card" style="border-left: 5px solid {quadrant['color']};">
                        <span class="quadrant-badge" style="background-color: {quadrant['color']}22; color: {quadrant['color']};">
                            {quadrant['emoji']} {quadrant['label']}
                        </span>
                        <p style="margin-top: 0.5rem;">
                            This track lands <strong>{intensity}</strong> in the
                            <strong>{quadrant['label']}</strong> region of the emotion
                            space. {quadrant['description']}
                        </p>
                        <p style="margin-bottom: 0;" class="small-muted">
                            {' &nbsp;·&nbsp; '.join(quadrant['tags'])}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### Emotional arc")
                st.plotly_chart(build_trajectory_figure(chunk_1_9[:, 0], chunk_1_9[:, 1]), use_container_width=True)
                st.caption(
                    f"Valence and arousal re-estimated on each ~{CHUNK_DURATION_SECONDS:.1f}s "
                    "chunk of the clip, showing how the predicted mood shifts across the song."
                )

            with res_col2:
                st.markdown("#### Where it lands on the emotion map")
                st.plotly_chart(
                    build_circumplex_figure(valence, arousal, quadrant, trajectory=chunk_1_9),
                    use_container_width=True,
                )
                st.caption(
                    "Follows Russell's circumplex model of affect, the standard reference "
                    "frame in music emotion recognition. Faint dots show each chunk; the "
                    "large dot is the overall prediction."
                )

            with st.expander("Raw per-chunk predictions"):
                chunk_df = pd.DataFrame(chunk_1_9, columns=["Valence", "Arousal"])
                chunk_df.index = [f"{i * CHUNK_DURATION_SECONDS:.1f}s" for i in range(len(chunk_df))]
                st.dataframe(chunk_df.style.format("{:.3f}"), use_container_width=True)

            with st.expander("How is this calculated?"):
                st.write(
                    f"""
                    The model outputs raw valence and arousal scores between 0 and 1
                    ({valence_raw:.3f} and {arousal_raw:.3f} for this track, averaged
                    across {NUM_CHUNKS} chunks), converted to the 1–9 DEAM annotation
                    scale via `raw * 8 + 1`. The quadrant label comes from where the
                    point falls relative to the midpoint (5.0) on each axis, following
                    Russell's circumplex model of affect.
                    """
                )

        except ValueError as e:
            st.error(
                f"**Couldn't process this file.** {e}\n\n"
                f"Try a clip that's at least ~{MIN_AUDIO_SECONDS:.0f} seconds long."
            )
        except Exception as e:
            st.error(f"Something went wrong while processing this file: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
else:
    st.markdown(f"""
    <div class="card">
        <b>How it works</b>
        <ol>
            <li>Upload a song — at least {MIN_AUDIO_SECONDS:.0f} seconds long.</li>
            <li>The audio is converted into a mel spectrogram using the exact same
                parameters, normalization, and chunking used during training.</li>
            <li>The AST model scores each ~{CHUNK_DURATION_SECONDS:.1f}s chunk, and the
                results are averaged into a single valence/arousal prediction and
                plotted on the emotion circumplex.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "Model: AST fine-tuned on DEAM · Metrics from held-out test split · "
    "Valence/arousal circumplex model of affect (Russell, 1980)"
)
