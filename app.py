# app.py
import streamlit as st
import yake
from rake_nltk import Rake
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import tempfile
import PyPDF2
import pandas as pd

# ---------------------------
# Setup (downloads once)
# ---------------------------
nltk.download("punkt")
nltk.download("stopwords")

# ---------------------------
# Extraction Functions
# ---------------------------
def yake_extract(text, top_n=15):
    kw_extractor = yake.KeywordExtractor(top=top_n, stopwords=None)
    kw = kw_extractor.extract_keywords(text)
    return [k for k, score in kw]

def rake_extract(text, top_n=15):
    r = Rake()
    r.extract_keywords_from_text(text)
    phrases = r.get_ranked_phrases()
    return phrases[:top_n]

def tfidf_extract(text, top_n=15):
    # simple TF-IDF on the single document: we split into sentences as "corpus"
    sentences = [s for s in text.split(".") if s.strip()]
    if not sentences:
        sentences = [text]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=500)
    X = vectorizer.fit_transform(sentences)
    # compute average tfidf across sentences for each term
    scores = X.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    term_scores = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return [t for t, s in term_scores[:top_n]]

def hybrid_extract(text, top_n=20):
    # combine many sources, score by occurrence and method-rank
    sources = []
    sources += yake_extract(text, top_n * 2)
    sources += rake_extract(text, top_n * 2)
    sources += tfidf_extract(text, top_n * 2)
    # scoring: earlier occurrences get slightly higher weight (rank-based)
    score = {}
    for src in sources:
        # normalize
        k = src.strip().lower()
        if not k:
            continue
        score[k] = score.get(k, 0) + 1
    # sort by score then alphabetically
    ranked = sorted(score.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, s in ranked][:top_n]

# ---------------------------
# Utility: PDF -> text
# ---------------------------
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception:
        return ""

# ---------------------------
# Streamlit UI + Styling
# ---------------------------
# ---------------------------
# Streamlit UI + Styling
# ---------------------------
st.set_page_config(page_title="Keyword Extractor Pro", layout="wide", page_icon="✨")

# Custom CSS for Premium Look
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }

    /* Title Styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(to right, #60a5fa, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.05em;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }

    /* Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }

    /* Keyword Tags */
    .kw-tag {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.4rem;
        border-radius: 12px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #e0e7ff;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .kw-tag:hover {
        background: rgba(99, 102, 241, 0.25);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(to right, #4f46e5, #7c3aed);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        opacity: 0.9;
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.4);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Input Fields */
    .stTextArea textarea {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 1px #818cf8 !important;
    }
    
    /* Remove footer */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown("<div class='main-title'>Keyword Extraction Pro</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Unlock insights with advanced NLP models</div>", unsafe_allow_html=True)

# Sidebar for Settings
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    method = st.selectbox(
        "Extraction Model",
        ["Hybrid (Recommended)", "YAKE", "RAKE", "TF-IDF"],
        help="Select the algorithm to extract keywords."
    )
    
    num = st.slider("Keyword Limit", 5, 100, 15, help="Maximum number of keywords to extract.")
    
    st.markdown("---")
    st.markdown("### 📥 Export Settings")
    download_name = st.text_input("Filename", value="keywords_export")
    
    st.markdown("---")
    st.info("💡 **Tip:** Hybrid mode combines multiple algorithms for the most robust results.")

# Main Content
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📄 Source Text")
    
    tab1, tab2 = st.tabs(["✍️ Paste Text", "📂 Upload File"])
    
    with tab1:
        text_input = st.text_area("Enter your text here...", height=350, placeholder="Paste your article, report, or document content here to extract keywords...")
    
    with tab2:
        uploaded = st.file_uploader("Upload PDF or Text file", type=["pdf", "txt", "md"])
        if uploaded is not None:
            if uploaded.type == "application/pdf":
                with st.spinner("Reading PDF..."):
                    text_from_pdf = extract_text_from_pdf(uploaded)
                    if text_from_pdf.strip():
                        st.success(f"✅ Loaded {len(text_from_pdf)} characters from PDF")
                        if not text_input.strip():
                            text_input = text_from_pdf
                    else:
                        st.error("Could not extract text. Please try another file.")
            else:
                bytes_data = uploaded.read()
                try:
                    text_from_file = bytes_data.decode("utf-8")
                    if not text_input.strip():
                        text_input = text_from_file
                except:
                    st.error("Error reading file encoding.")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🚀 Actions")
    
    if st.button("Extract Keywords", use_container_width=True):
        doc_text = text_input.strip()
        
        if not doc_text:
            st.warning("⚠️ Please provide some text to analyze.")
        else:
            with st.spinner(f"Running {method} extraction..."):
                try:
                    if method == "YAKE":
                        kws = yake_extract(doc_text, top_n=num)
                    elif method == "RAKE":
                        kws = rake_extract(doc_text, top_n=num)
                    elif method == "TF-IDF":
                        kws = tfidf_extract(doc_text, top_n=num)
                    else:  # Hybrid
                        kws = hybrid_extract(doc_text, top_n=num)
                    
                    st.session_state['kws'] = kws
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # Display Results if available
    if 'kws' in st.session_state and st.session_state['kws']:
        st.markdown("---")
        st.markdown("### 🎯 Results")
        
        # Display as pills/tags
        tags_html = "".join([f"<span class='kw-tag'>{k}</span>" for k in st.session_state['kws']])
        st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # CSV Download
        df = pd.DataFrame({"Keyword": st.session_state['kws']})
        csv = df.to_csv(index=False).encode("utf-8")
        
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"{download_name}.csv",
            mime="text/csv",
            use_container_width=True
        )
            
    st.markdown("</div>", unsafe_allow_html=True)
