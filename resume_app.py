import streamlit as st
import fitz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Download stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load Sentence Transformer
model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

# Keywords for rule-based backup prediction
roles_keywords = {
    "data analyst": ["excel", "sql", "tableau", "power bi", "statistics", "dashboard"],
    "data scientist": ["python", "machine learning", "model", "regression", "nlp"],
    "business analyst": ["business", "requirement", "stakeholder", "report", "gap"],
    "ml engineer": ["tensorflow", "pytorch", "deep learning", "deployment", "neural"],
    "java developer": ["java", "spring", "hibernate", "microservices"],
    "devops engineer": ["docker", "kubernetes", "ci/cd", "aws", "pipeline"],
    "support engineer": ["ticket", "troubleshoot", "customer", "helpdesk"],
    "mechanical engineer": ["autocad", "solidworks", "design", "cad"]
}

# === TEXT CLEANING ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return " ".join([word for word in text.split() if word not in stop_words])

# === TEXT EXTRACTORS ===
def extract_text_from_pdf(uploaded_file):
    text = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return clean_text(text)

def extract_text_from_txt(uploaded_file):
    return clean_text(uploaded_file.read().decode("utf-8"))

# === ROLE PREDICTION (TF-IDF based + Keywords backup) ===
def predict_role(text, roles_keywords):
    vectorizer = TfidfVectorizer()
    documents = [text] + [" ".join(keywords) for keywords in roles_keywords.values()]
    tfidf_matrix = vectorizer.fit_transform(documents)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    best_match_index = scores.argmax()
    best_score = scores[best_match_index]

    # If too low score, fallback to keyword count
    if best_score < 0.1:
        keyword_scores = {
            role: sum(word in text for word in keywords)
            for role, keywords in roles_keywords.items()
        }
        return max(keyword_scores, key=keyword_scores.get)
    return list(roles_keywords.keys())[best_match_index]

# === SIMILARITY SCORE ===
def compare_with_jd(resume_text, jd_text, resume_role, jd_role):
    # Use hybrid method: semantic similarity + TF-IDF cosine
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    sim_score = util.pytorch_cos_sim(resume_emb, jd_emb).item()

    # Role match bonus
    if resume_role.lower() == jd_role.lower():
        sim_score += 0.1

    sim_score = min(sim_score, 1.0)
    return round(sim_score * 100, 2)

# === STREAMLIT UI ===
st.set_page_config(page_title="Smart Resume Matcher", layout="centered")
st.title("🎯 Resume to JD Matcher - Enhanced AI")

st.markdown("Upload a **resume (PDF)** and multiple **job descriptions (TXT)**. We'll predict the best role and show the most matching JD with improved accuracy!")

uploaded_resume = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
if uploaded_resume:
    resume_text = extract_text_from_pdf(uploaded_resume)
    resume_role = predict_role(resume_text, roles_keywords)
    st.success(f"🧾 Resume Role Prediction: **{resume_role.title()}**")

    jd_files = st.file_uploader("📑 Upload Job Descriptions (TXT files)", type=["txt"], accept_multiple_files=True)
    if jd_files:
        best_score = 0
        best_jd_text = ""
        best_index = -1

        for idx, jd_file in enumerate(jd_files):
            jd_text = extract_text_from_txt(jd_file)
            jd_role = predict_role(jd_text, roles_keywords)
            score = compare_with_jd(resume_text, jd_text, resume_role, jd_role)

            st.subheader(f"📋 JD {idx+1} - {jd_file.name}")
            st.write(f"Predicted JD Role: **{jd_role.title()}**")
            st.metric("🔍 Match Score", f"{score}%")

            if score > best_score:
                best_score = score
                best_jd_text = jd_text
                best_index = idx

            if score >= 80:
                st.success("✅ Strong match!")
            elif score >= 60:
                st.warning("🛠 Decent match, consider tweaking resume.")
            else:
                st.error("❌ Not a good fit.")

        if best_jd_text:
            st.markdown("---")
            st.subheader(f"🏆 Best Match: JD {best_index + 1}")
            st.text_area("Best Job Description", best_jd_text, height=200)
