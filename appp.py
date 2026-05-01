import streamlit as st
import PyPDF2
import re
import io
import openai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Screener",
    layout="wide"
)

st.title("AI Resume Screener")

st.write(
    "Paste a job description, upload resume PDFs, and instantly see every "
    "candidate ranked by how well they match — with optional AI explanations."
)
st.divider()

# extract text from pdf
def extract_text_from_pdf(file_byte):
    reader=PyPDF2.PdfReader(io.BytesIO(file_byte))
    text=""
    for page in reader.pages:
        exratcted=page.extract_text()
        if exratcted:
            text+=exratcted+"\n"
    return text.strip()

#cleaning the text
def clean_text(text):
    text=re.sub(r'http\S+',' ',text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# score resume
def score_resume(job_description,resumes):
    jd_clean=clean_text(job_description)
    resume_names=list(resumes.keys())
    resume_texts=[clean_text(resumes[name]) for name in resume_names]
    all_docs=[jd_clean]+resume_texts
    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2))
    tfidf_matrix=vectorizer.fit_transform(all_docs)
    jd_vector=tfidf_matrix[0]
    resume_vectors=tfidf_matrix[1:]
    similarities=cosine_similarity(jd_vector,resume_vectors)[0]
    scores={}
    for i, name in enumerate(resume_names):
        scores[name]=round(float(similarities[i])*100,1)
    return scores


#get ai explaination
def get_gpt_explaination(job_description,resume_text,candidate_name,score,api_key):
    if not api_key:
        return None
    try:
        client=openai.OpenAI(api_key=api_key)
        prompt = f"""You are an expert HR recruiter reviewing a candidate.

Job Description:
{job_description[:1500]}

Resume of {candidate_name} (match score: {score}/100):
{resume_text[:2000]}

Write exactly 3 bullet points explaining why this candidate fits or does not fit.
Be specific — mention actual skills from the resume compared to the job description.
Start each bullet with "Fits:" or "Gap:" depending on whether it is a strength or weakness.
Keep each bullet under 25 words."""
        response=client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            max_tokens=250,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Clould not get gpt explaination;{str(e)}"

#score lable
def score_label(score):
    if score>=60:
        return "Strong match"
    elif score>=35:
        return "partial match"
    else:
        return "Weak match"
    
#ui
col_left,col_right=st.columns([1,1],gap="large")
with col_left:
    job_description=st.text_area(
        "Paste the job description here",
        height=320,
        placeholder=("paste job description"),
        label_visibility="collapsed"
    )
with col_right:
    uploaded_files = st.file_uploader(
        "Upload resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    api_key = st.text_input(
        "OpenAI key",
        type="password",
        placeholder="sk-...   Leave blank to skip AI explanations",
        label_visibility="collapsed"
    )

#button
if st.button("Screen Resumes", type="primary", use_container_width=True):
        if not job_description.strip():
            st.error("Please paste a job description before screening.")
            st.stop()
        resumes = {}
        with st.spinner("Reading resumes..."):
            for f in uploaded_files:
                try:
                    text = extract_text_from_pdf(f.read())
                    if text:
                        resumes[f.name] = text
                    else:
                        st.warning(f"Could not read text from {f.name}...")
                except Exception as e:
                    st.warning(f"Error reading {f.name}: {e}. Skipping.")
        with st.spinner("Calculating match scores..."):
            scores = score_resume(job_description, resumes)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)  
        for rank, (filename, score) in enumerate(ranked, start=1):
            display_name = (
            filename
            .replace(".pdf", "")
            .replace("_", " ")
            .replace("-", " ")
            .title()
        )
        col_rank, col_name, col_score = st.columns([0.08, 0.67, 0.25])
        with col_rank:
            st.markdown(
                f"<div style='font-size:22px;font-weight:800;color:#4f9cf9;"
                f"padding-top:6px'>#{rank}</div>",
                unsafe_allow_html=True
            )
        with col_score:
            color = "#10b981" if score >= 60 else "#f59e0b" if score >= 35 else "#ef4444"

            strong_count =sum(1 for s in scores.values() if s >= 60)
            partial_count = sum(1 for s in scores.values()if 35 <= s < 60)
            avg_score = (sum(scores.values()) / len(scores)if scores else 0)

            col_a, col_b, col_c, col_d = st.columns(4)
            
            col_a.metric("Total Screened",  len(ranked))
            col_b.metric("Strong Matches",  strong_count)
            col_c.metric("Partial Matches", partial_count)
            col_d.metric("Average Score",   f"{avg_score:.1f}")