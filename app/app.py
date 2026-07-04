import os
import re
import io
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

matplotlib.use("Agg")

# ───────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="ABSA NER — Review Grab",
    layout="wide",
)

# ───────────────────────────────────────────────
# CUSTOM CSS
# ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Font & background */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Header utama */
    .app-header {
        background: #1E3A5F;
        color: white;
        padding: 28px 36px;
        border-radius: 8px;
        margin-bottom: 24px;
    }
    .app-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        color: white;
    }
    .app-header p {
        font-size: 0.88rem;
        color: #CBD5E1;
        margin: 0;
    }

    /* Metric card */
    .metric-row {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
    }
    .metric-card {
        flex: 1;
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-card .val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    .metric-card .lbl {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Section header */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1E293B;
        border-left: 4px solid #1E3A5F;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    /* NER token highlight */
    .ner-container {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        line-height: 2.4;
        min-height: 60px;
    }
    .ner-token {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 5px;
        margin: 2px;
        font-size: 0.88rem;
        font-weight: 500;
        color: white;
    }
    .ner-label {
        font-size: 0.65rem;
        opacity: 0.85;
        margin-left: 4px;
        font-weight: 400;
    }

    /* Sentimen badge */
    .badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-positif  { background: #DCFCE7; color: #166534; }
    .badge-negatif  { background: #FEE2E2; color: #991B1B; }
    .badge-netral   { background: #FEF9C3; color: #854D0E; }

    /* Tabel ringkas */
    .info-box {
        background: #F1F5F9;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.85rem;
        color: #334155;
        margin-top: 8px;
    }

    /* Divider tipis */
    hr { border: none; border-top: 1px solid #E2E8F0; margin: 28px 0; }

    /* Sembunyikan header default streamlit */
    #MainMenu, footer, header { visibility: hidden; }

    /* Tombol */
    .stButton > button {
        background: #1E3A5F;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 24px;
        font-weight: 600;
        font-size: 0.88rem;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: #2D5186;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# NLTK DOWNLOAD
# ───────────────────────────────────────────────
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# ───────────────────────────────────────────────
# PATH
# ───────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

# ───────────────────────────────────────────────
# SLANG & ASPEK
# ───────────────────────────────────────────────
SLANG_DICT = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak",
    "udah": "sudah", "udh": "sudah", "dah": "sudah",
    "bgt": "banget", "bget": "banget",
    "yg": "yang", "dgn": "dengan", "utk": "untuk",
    "krn": "karena", "karna": "karena",
    "app": "aplikasi", "apk": "aplikasi",
    "dr": "dari", "lg": "lagi", "jg": "juga",
}

ASPECT_KEYWORDS = {
    "driver":   ["driver", "supir", "pengemudi"],
    "aplikasi": ["aplikasi", "app", "apk", "sistem"],
    "harga":    ["harga", "ongkir", "tarif", "biaya"],
    "layanan":  ["layan", "layanan", "pelayanan", "respon", "cs"],
    "keamanan": ["aman", "keamanan", "safety"],
}

# ───────────────────────────────────────────────
# SASTRAWI
# ───────────────────────────────────────────────
@st.cache_resource
def get_sastrawi():
    stemmer   = StemmerFactory().create_stemmer()
    base_sw   = set(StopWordRemoverFactory().get_stop_words())
    keep      = {"tidak", "jangan", "kurang", "bukan"}
    stopwords = base_sw - keep
    return stemmer, stopwords

# ───────────────────────────────────────────────
# PREPROCESSING
# ───────────────────────────────────────────────
def preprocess(text: str) -> str:
    stemmer, stopwords = get_sastrawi()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [SLANG_DICT.get(w, w) for w in text.split()]
    tokens = word_tokenize(" ".join(words))
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

def detect_aspect(text: str) -> str:
    tokens = set(text.lower().split())
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(kw in tokens for kw in keywords):
            return aspect
    return "lainnya"

# ───────────────────────────────────────────────
# LOAD MODEL
# ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model IndoBERT...")
def load_models():
    try:
        tok_absa  = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "indobert_absa_tokenizer"))
        mod_absa  = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, "indobert_absa_model"))
        tok_ner   = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "indobert_ner_tokenizer"), use_fast=True)
        mod_ner   = AutoModelForTokenClassification.from_pretrained(os.path.join(MODEL_DIR, "indobert_ner_model"))
        mod_absa.eval()
        mod_ner.eval()
        return tok_absa, mod_absa, tok_ner, mod_ner
    except Exception as e:
        st.error(f"Model tidak dapat dimuat: {e}")
        st.stop()

# ───────────────────────────────────────────────
# PREDICT
# ───────────────────────────────────────────────
LABEL_MAP    = {0: "negatif", 1: "netral", 2: "positif"}
NER_ID2LABEL = {0: "B-ASPECT", 1: "B-ORG", 2: "B-PROD", 3: "O"}

def predict_absa(clean_text, aspect, tokenizer, model):
    pair   = f"{clean_text} [SEP] {aspect}"
    inputs = tokenizer(pair, return_tensors="pt", truncation=True,
                       padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs    = torch.softmax(logits, dim=1).squeeze().tolist()
    pred_idx = int(torch.argmax(torch.tensor(probs)).item())
    return {
        "label": LABEL_MAP[pred_idx],
        "confidence": {
            "negatif": probs[0],
            "netral":  probs[1],
            "positif": probs[2],
        },
    }

def predict_ner(clean_text, tokenizer, model):
    words = clean_text.split()
    if not words:
        return []
    inputs   = tokenizer(words, is_split_into_words=True, return_tensors="pt",
                         truncation=True, padding=True, max_length=128)
    word_ids = inputs.word_ids()
    with torch.no_grad():
        logits = model(**inputs).logits
    preds  = torch.argmax(logits, dim=2).squeeze().tolist()
    if not isinstance(preds, list):
        preds = [preds]
    results, seen = [], set()
    for i, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        label = NER_ID2LABEL.get(preds[i], "O")
        results.append((words[wid], label))
    return results

# ───────────────────────────────────────────────
# NER HTML
# ───────────────────────────────────────────────
NER_COLORS = {
    "B-ASPECT": "#2563EB",
    "B-ORG":    "#059669",
    "B-PROD":   "#D97706",
}

def render_ner_html(ner_results):
    parts = []
    for token, label in ner_results:
        if label in NER_COLORS:
            color = NER_COLORS[label]
            short = label.replace("B-", "")
            parts.append(
                f'<span class="ner-token" style="background:{color};">'
                f'{token}<span class="ner-label">[{short}]</span></span>'
            )
        else:
            parts.append(
                f'<span style="margin:2px; display:inline-block; '
                f'font-size:0.88rem; color:#374151;">{token}</span>'
            )
    return f'<div class="ner-container">{" ".join(parts)}</div>'

# ───────────────────────────────────────────────
# SESSION STATE
# ───────────────────────────────────────────────
for key in ["hasil_analisis", "hasil_csv"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ───────────────────────────────────────────────
# LOAD MODEL
# ───────────────────────────────────────────────
tokenizer_absa, model_absa, tokenizer_ner, model_ner = load_models()

# ═══════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <h1>Aspect-Based Sentiment Analysis dan Named Entity Recognition</h1>
    <p>
        Analisis ulasan pengguna aplikasi Grab (Google Play Store) menggunakan 
        model IndoBERT fine-tuned untuk klasifikasi sentimen berbasis aspek dan 
        pengenalan entitas bernama dalam Bahasa Indonesia.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="metric-row">
    <div class="metric-card">
        <div class="val">5.000</div>
        <div class="lbl">Total Data Scraping</div>
    </div>
    <div class="metric-card">
        <div class="val">3.259</div>
        <div class="lbl">Total Data ABSA</div>
    </div>
    <div class="metric-card">
        <div class="val">0.8549</div>
        <div class="lbl">ABSA Weighted F1</div>
    </div>
    <div class="metric-card">
        <div class="val">0.9500</div>
        <div class="lbl">NER Macro F1</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# SECTION 1 — ANALISIS TEKS
# ═══════════════════════════════════════════════
st.markdown('<div class="section-title">Analisis Teks</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#64748B; font-size:0.88rem; margin-bottom:16px;">'
    'Masukkan ulasan aplikasi Grab dalam Bahasa Indonesia. '
    'Sistem akan mengidentifikasi aspek, sentimen, dan entitas yang terkandung dalam teks.</p>',
    unsafe_allow_html=True
)

raw_text = st.text_area(
    "Teks Ulasan",
    placeholder="Contoh: Drivernya sangat ramah dan tepat waktu, tapi tarifnya mahal sekali...",
    height=110,
    label_visibility="collapsed",
)

if st.button("Analisis"):
    if not raw_text.strip():
        st.warning("Teks ulasan tidak boleh kosong.")
    else:
        with st.spinner("Memproses..."):
            clean   = preprocess(raw_text)
            aspect  = detect_aspect(clean)
            ner_res = predict_ner(clean, tokenizer_ner, model_ner)
            absa    = predict_absa(clean, aspect, tokenizer_absa, model_absa)
            st.session_state["hasil_analisis"] = {
                "raw": raw_text, "clean": clean,
                "aspect": aspect, "ner_results": ner_res,
                "ner_html": render_ner_html(ner_res), "absa": absa,
            }

# Tampilkan hasil
if st.session_state["hasil_analisis"]:
    h = st.session_state["hasil_analisis"]

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        # NER
        st.markdown('<div class="section-title" style="font-size:0.9rem;">Hasil Named Entity Recognition</div>', unsafe_allow_html=True)
        st.markdown(h["ner_html"], unsafe_allow_html=True)

        # Legenda
        st.markdown("""
        <div style="margin-top:10px; font-size:0.78rem; color:#64748B; display:flex; gap:16px;">
            <span><span style="background:#2563EB; color:white; padding:2px 8px; 
            border-radius:4px;">ASPECT</span> Aspek ulasan</span>
            <span><span style="background:#059669; color:white; padding:2px 8px; 
            border-radius:4px;">ORG</span> Organisasi</span>
            <span><span style="background:#D97706; color:white; padding:2px 8px; 
            border-radius:4px;">PROD</span> Produk</span>
        </div>
        """, unsafe_allow_html=True)

        # Tabel token
        if h["ner_results"]:
            ner_df = pd.DataFrame(h["ner_results"], columns=["Token", "Label"])
            st.dataframe(ner_df, use_container_width=True, hide_index=True, height=200)

    with col_r:
        # ABSA
        st.markdown('<div class="section-title" style="font-size:0.9rem;">Hasil Aspect-Based Sentiment Analysis</div>', unsafe_allow_html=True)

        sentiment  = h["absa"]["label"]
        confidence = h["absa"]["confidence"]
        max_conf   = max(confidence.values())

        st.markdown(f"""
        <div class="info-box" style="margin-bottom:16px;">
            <table style="width:100%; border-collapse:collapse;">
                <tr>
                    <td style="color:#64748B; font-size:0.82rem; width:40%;">Aspek Terdeteksi</td>
                    <td style="font-weight:600; color:#1E293B;">{h["aspect"].capitalize()}</td>
                </tr>
                <tr>
                    <td style="color:#64748B; font-size:0.82rem; padding-top:8px;">Sentimen</td>
                    <td style="padding-top:8px;">
                        <span class="badge badge-{sentiment}">{sentiment.upper()}</span>
                    </td>
                </tr>
                <tr>
                    <td style="color:#64748B; font-size:0.82rem; padding-top:8px;">Confidence</td>
                    <td style="font-weight:600; color:#1E293B; padding-top:8px;">{max_conf:.2%}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar per kelas
        st.markdown('<p style="font-size:0.82rem; color:#64748B; margin-bottom:8px;">Distribusi Confidence Score</p>', unsafe_allow_html=True)
        for lbl, score in confidence.items():
            st.progress(float(score), text=f"{lbl}: {score:.2%}")

st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# SECTION 2 — UPLOAD CSV
# ═══════════════════════════════════════════════
st.markdown('<div class="section-title">Analisis Massal (Upload CSV)</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#64748B; font-size:0.88rem; margin-bottom:16px;">'
    'Upload file CSV dengan kolom <code>text</code> untuk analisis sentimen dan entitas secara massal.</p>',
    unsafe_allow_html=True
)

uploaded = st.file_uploader("Upload file CSV", type=["csv"], label_visibility="collapsed")

if uploaded:
    df_up = pd.read_csv(uploaded)
    if "text" not in df_up.columns:
        st.error("File CSV harus memiliki kolom 'text'.")
    else:
        st.markdown(f'<p style="font-size:0.82rem; color:#64748B;">Preview data ({len(df_up)} baris)</p>', unsafe_allow_html=True)
        st.dataframe(df_up.head(), use_container_width=True, hide_index=True)

        if st.button("Proses Semua Data"):
            results = []
            bar = st.progress(0, text="Memproses data...")
            total = len(df_up)
            for i, row in df_up.iterrows():
                raw  = str(row["text"])
                cln  = preprocess(raw)
                asp  = detect_aspect(cln)
                res  = predict_absa(cln, asp, tokenizer_absa, model_absa)
                conf = max(res["confidence"].values())
                results.append({
                    "Teks": raw,
                    "Aspek": asp,
                    "Sentimen": res["label"],
                    "Confidence": round(conf, 4),
                })
                bar.progress((i + 1) / total, text=f"Memproses {i+1}/{total}...")
            bar.empty()
            st.session_state["hasil_csv"] = pd.DataFrame(results)

if st.session_state["hasil_csv"] is not None:
    df_res = st.session_state["hasil_csv"]
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="section-title" style="font-size:0.9rem;">Hasil Prediksi</div>', unsafe_allow_html=True)
        st.dataframe(df_res, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Hasil CSV",
            data=df_res.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi_absa.csv",
            mime="text/csv",
        )

    with col_r:
        st.markdown('<div class="section-title" style="font-size:0.9rem;">Distribusi Hasil</div>', unsafe_allow_html=True)

        sent_counts = df_res["Sentimen"].value_counts()
        colors_map  = {"positif": "#059669", "negatif": "#DC2626", "netral": "#D97706"}
        pie_colors  = [colors_map.get(s, "#94A3B8") for s in sent_counts.index]

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

        axes[0].pie(sent_counts, labels=sent_counts.index,
                    autopct="%1.1f%%", colors=pie_colors, startangle=90,
                    textprops={"fontsize": 9})
        axes[0].set_title("Distribusi Sentimen", fontsize=10, fontweight="bold", pad=10)

        asp_counts = df_res["Aspek"].value_counts()
        bars = axes[1].barh(asp_counts.index, asp_counts.values, color="#2563EB", height=0.5)
        axes[1].bar_label(bars, padding=4, fontsize=8)
        axes[1].set_xlabel("Jumlah", fontsize=8)
        axes[1].set_title("Distribusi Aspek", fontsize=10, fontweight="bold")
        axes[1].tick_params(labelsize=8)
        axes[1].invert_yaxis()
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        fig.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close(fig)

st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# SECTION 3 — EVALUASI MODEL
# ═══════════════════════════════════════════════
st.markdown('<div class="section-title">Evaluasi Model</div>', unsafe_allow_html=True)

tab_absa, tab_ner = st.tabs(["ABSA", "NER"])

with tab_absa:
    st.markdown("""
    <div class="info-box" style="margin-bottom:20px;">
        <strong>Model:</strong> IndoBERT fine-tuned (indobenchmark/indobert-base-p1) — 
        BertForSequenceClassification<br>
        <strong>Pendekatan:</strong> Sentence-pair classification (Teks + Aspek) dengan token [SEP] sebagai pemisah<br>
        <strong>Penanganan Imbalance:</strong> Random Oversampling pada kelas Netral (77 → 350 sampel latih)<br>
        <strong>Split:</strong> 80% train (2.607) / 20% test (652) dengan stratifikasi
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<p style="font-size:0.88rem; font-weight:600; color:#1E293B; margin-bottom:10px;">Classification Report</p>', unsafe_allow_html=True)
        report_data = {
            "Kelas":     ["negatif", "netral", "positif", "weighted avg"],
            "Precision": [0.83, 0.00, 0.91, 0.85],
            "Recall":    [0.88, 0.00, 0.90, 0.86],
            "F1-Score":  [0.85, 0.00, 0.90, 0.85],
            "Support":   [275, 19, 358, 652],
        }
        st.dataframe(
            pd.DataFrame(report_data).style.format({
                "Precision": "{:.2f}",
                "Recall": "{:.2f}",
                "F1-Score": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("""
        <div class="info-box" style="border-left: 3px solid #F59E0B; background:#FFFBEB;">
            <strong>Catatan:</strong> F1-Score kelas <em>netral</em> bernilai 0.00 
            disebabkan oleh jumlah sampel uji yang sangat terbatas (19 dari 652 data uji). 
            Ketidakseimbangan distribusi kelas pada data uji menyebabkan model tidak 
            mampu mengenali pola sentimen netral secara konsisten.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<p style="font-size:0.88rem; font-weight:600; color:#1E293B; margin-bottom:10px;">Confusion Matrix</p>', unsafe_allow_html=True)
        cm     = np.array([[242, 7, 26], [12, 0, 7], [33, 2, 323]])
        labels = ["negatif", "netral", "positif"]
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    ax=ax_cm, linewidths=0.5, linecolor="#E2E8F0",
                    annot_kws={"size": 11})
        ax_cm.set_xlabel("Prediksi", fontsize=9)
        ax_cm.set_ylabel("Aktual", fontsize=9)
        ax_cm.set_title("Confusion Matrix — ABSA IndoBERT", fontsize=9, fontweight="bold")
        ax_cm.tick_params(labelsize=8)
        fig_cm.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

with tab_ner:
    st.markdown("""
    <div class="info-box" style="margin-bottom:20px;">
        <strong>Model:</strong> IndoBERT fine-tuned — BertForTokenClassification<br>
        <strong>Pendekatan:</strong> Token-level sequence labeling dengan skema BIO<br>
        <strong>Entitas:</strong> B-ASPECT, B-ORG, B-PROD, O<br>
        <strong>Split:</strong> 80% train (3.812 kalimat) / 20% test (953 kalimat)
    </div>
    """, unsafe_allow_html=True)

    col_c, col_d = st.columns(2, gap="large")

    with col_c:
        st.markdown('<p style="font-size:0.88rem; font-weight:600; color:#1E293B; margin-bottom:10px;">Classification Report per Entitas</p>', unsafe_allow_html=True)
        ner_data = {
            "Entitas":   ["B-ASPECT", "B-ORG", "B-PROD", "O", "weighted avg"],
            "Precision": [0.93, 0.94, 0.75, 1.00, 1.00],
            "Recall":    [1.00, 1.00, 1.00, 0.99, 0.99],
            "F1-Score":  [0.96, 0.97, 0.86, 1.00, 1.00],
            "Support":   [303, 149, 6, 6415, 6873],
        }
        st.dataframe(
            pd.DataFrame(ner_data).style.format({
                "Precision": "{:.2f}",
                "Recall": "{:.2f}",
                "F1-Score": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("""
        <div class="info-box" style="border-left: 3px solid #3B82F6; background:#EFF6FF;">
            Label <strong>O</strong> mendominasi 93.3% token (6.415/6.873), 
            sehingga weighted F1 cenderung tinggi. Evaluasi yang lebih relevan 
            adalah macro F1 per entitas. Label <strong>B-PROD</strong> hanya 
            memiliki 6 sampel uji sehingga metriknya belum representatif secara statistik.
        </div>
        """, unsafe_allow_html=True)

    with col_d:
        st.markdown('<p style="font-size:0.88rem; font-weight:600; color:#1E293B; margin-bottom:10px;">F1-Score per Entitas</p>', unsafe_allow_html=True)
        ner_labels = ["B-ASPECT", "B-ORG", "B-PROD", "O"]
        ner_f1     = [0.96, 0.97, 0.86, 1.00]
        bar_colors = ["#2563EB", "#059669", "#D97706", "#94A3B8"]
        fig_ner, ax_ner = plt.subplots(figsize=(4.5, 3.5))
        bars = ax_ner.bar(ner_labels, ner_f1, color=bar_colors,
                          width=0.5, edgecolor="white", linewidth=1)
        ax_ner.bar_label(bars, fmt="%.2f", padding=4, fontsize=9, fontweight="bold")
        ax_ner.set_ylim(0, 1.15)
        ax_ner.set_ylabel("F1-Score", fontsize=9)
        ax_ner.set_title("F1-Score per Entitas NER", fontsize=9, fontweight="bold")
        ax_ner.axhline(y=1.0, linestyle="--", color="#94A3B8", alpha=0.5, linewidth=1)
        ax_ner.spines["top"].set_visible(False)
        ax_ner.spines["right"].set_visible(False)
        ax_ner.tick_params(labelsize=8)
        fig_ner.tight_layout()
        st.pyplot(fig_ner)
        plt.close(fig_ner)

st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding:16px 0; font-size:0.78rem; color:#94A3B8;">
    Alvin Deo Ardiansyah &nbsp;|&nbsp; A11.2023.15072 &nbsp;|&nbsp;
    Pemrosesan Bahasa Alami Berbasis Teks &nbsp;|&nbsp;
    Junta Zeniarja, M.Kom &nbsp;|&nbsp;
    Universitas Dian Nuswantoro
</div>
""", unsafe_allow_html=True)