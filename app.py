# =============================================================================
# AI RESUME SCREENER — Week 1 Project
# =============================================================================
# What this app does:
#   1. You paste a job description
#   2. You upload multiple resume PDFs
#   3. It scores each resume 0-100 based on how well it matches the job
#   4. It ranks all candidates from best to worst
#   5. (Optional) Add OpenAI key → GPT writes 3 bullet points per candidate
#      explaining exactly why they fit or don't
#
# Core technique: TF-IDF + Cosine Similarity
#   - No pre-trained model file needed
#   - Everything is computed fresh each time someone uploads resumes
# =============================================================================

import streamlit as st      # builds the web UI
import PyPDF2               # reads text out of PDF files
import re                   # regular expressions — used for cleaning text
import io                   # lets us treat raw bytes as a file object
import openai               # OpenAI Python SDK — for GPT explanations

# TF-IDF and cosine similarity come from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# PAGE SETUP
# =============================================================================

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


# =============================================================================
# STEP 1 — EXTRACT TEXT FROM A PDF
# =============================================================================
# PyPDF2 reads the PDF page by page and pulls out the raw text.
# Some PDFs are scanned images — those return empty strings. We skip those.

def extract_text_from_pdf(file_bytes):
    """
    Takes raw bytes of a PDF file.
    Returns all the text inside as one big string.
    """
    # io.BytesIO wraps the raw bytes so PyPDF2 can read it like a file
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))

    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:               # some pages return None — skip those
            text += extracted + "\n"

    return text.strip()


# =============================================================================
# STEP 2 — CLEAN THE TEXT
# =============================================================================
# Raw resume text is messy — URLs, symbols, extra spaces.
# We strip all of that so TF-IDF only sees meaningful words.

def clean_text(text):
    """
    Removes URLs, social handles, punctuation, and extra whitespace.
    Returns clean plain text.
    """
    text = re.sub(r'http\S+', ' ', text)    # remove URLs
    text = re.sub(r'@\S+',   ' ', text)    # remove @mentions
    text = re.sub(r'[^\w\s]',' ', text)    # remove punctuation
    text = re.sub(r'\s+',    ' ', text)    # collapse multiple spaces
    return text.strip()


# =============================================================================
# STEP 3 — SCORE RESUMES USING TF-IDF + COSINE SIMILARITY
# =============================================================================
#
# HOW TF-IDF WORKS:
# Imagine you have 5 resumes and a job description.
# TF-IDF converts each of these 6 documents into a list of numbers (a vector).
# Each number represents how important a word is in that document.
#
# TF = how often a word appears in THIS document
# IDF = how rare that word is ACROSS ALL documents
#
# Example:
#   The word "Python" appears 10 times in a resume AND is rare across all docs
#   → high TF-IDF score → Python gets a big number in that resume's vector
#
#   The word "experience" appears everywhere in every doc
#   → low IDF → experience gets a small number in every vector
#
# This means technical keywords like "PyTorch", "Kubernetes", "XGBoost"
# get weighted much higher than generic words like "worked" or "team".
#
# HOW COSINE SIMILARITY WORKS:
# Once every document is a vector, we measure the ANGLE between two vectors.
# Small angle → similar content → high score (close to 1.0)
# Large angle → different content → low score (close to 0.0)
# We multiply by 100 to get a 0-100 score.
#
# WHY COSINE AND NOT JUST COUNTING KEYWORDS?
# Cosine similarity is length-independent. A 1-page resume and a 3-page resume
# with the same skills score the same. Pure keyword counting would unfairly
# reward longer resumes.

def score_resumes(job_description, resumes):
    """
    job_description: string — the full JD text
    resumes: dict of {filename: resume_text}

    Returns: dict of {filename: score_0_to_100}
    """

    # Clean both the JD and all resumes
    jd_clean = clean_text(job_description)
    resume_names = list(resumes.keys())
    resume_texts = [clean_text(resumes[name]) for name in resume_names]

    # Put JD first, then all resumes — TF-IDF needs to see all docs at once
    # so it can calculate IDF (how rare each word is across ALL documents)
    all_docs = [jd_clean] + resume_texts

    # Build the TF-IDF matrix
    # stop_words='english' removes words like "the", "and", "is" automatically
    # ngram_range=(1,2) means it looks at single words AND pairs of words
    #   so "machine learning" is treated as one feature, not two separate ones
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    # tfidf_matrix shape: (number_of_docs, number_of_unique_words)
    # Row 0 = job description vector
    # Rows 1 onwards = resume vectors

    jd_vector      = tfidf_matrix[0]      # job description row
    resume_vectors = tfidf_matrix[1:]     # all resume rows

    # cosine_similarity returns a 2D array
    # [0] gives us a flat list of one score per resume
    similarities = cosine_similarity(jd_vector, resume_vectors)[0]

    # Build result dict: multiply by 100 to get 0-100 score
    scores = {}
    for i, name in enumerate(resume_names):
        scores[name] = round(float(similarities[i]) * 100, 1)

    return scores


# =============================================================================
# STEP 4 — GPT EXPLANATION (optional)
# =============================================================================
# If the user provides an OpenAI API key, we send the JD + resume to GPT
# and ask it to write 3 bullet points explaining the fit or gaps.
# This turns a number (score: 67) into something a recruiter can act on.

def get_gpt_explanation(job_description, resume_text, candidate_name, score, api_key):
    """
    Calls GPT-3.5 and returns 3 bullet points about this candidate's fit.
    Returns None if no API key is provided.
    """
    if not api_key:
        return None

    try:
        client = openai.OpenAI(api_key=api_key)

        # We limit the text lengths to keep the prompt within token limits
        # GPT-3.5 has a context window — sending 10-page resumes would fail
        prompt = f"""You are an expert HR recruiter reviewing a candidate.

Job Description:
{job_description[:1500]}

Resume of {candidate_name} (match score: {score}/100):
{resume_text[:2000]}

Write exactly 3 bullet points explaining why this candidate fits or does not fit.
Be specific — mention actual skills from the resume compared to the job description.
Start each bullet with "Fits:" or "Gap:" depending on whether it is a strength or weakness.
Keep each bullet under 25 words."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.3      # lower temperature = more focused, less creative
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Could not get GPT explanation: {str(e)}"


# =============================================================================
# STEP 5 — HELPER: WHAT COLOR TO SHOW THE SCORE IN
# =============================================================================

def score_label(score):
    """Returns a human-readable label based on the score."""
    if score >= 60:
        return "Strong match"
    elif score >= 35:
        return "Partial match"
    else:
        return "Weak match"


# =============================================================================
# THE UI — Everything below is what the user sees
# =============================================================================

# Two columns side by side: left for JD, right for file upload + API key
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Job Description")
    job_description = st.text_area(
        "Paste the full job description here",
        height=320,
        placeholder=(
            "Example:\n\n"
            "We are looking for a Python Developer with experience in "
            "Django, REST APIs, PostgreSQL, and deploying applications "
            "on AWS. Must have 2+ years of backend development experience..."
        ),
        label_visibility="collapsed"
    )

with col_right:
    st.subheader("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,   # lets the user pick multiple files at once
        label_visibility="collapsed"
    )

    st.subheader("OpenAI API Key (optional)")
    api_key = st.text_input(
        "OpenAI key",
        type="password",              # hides the key as the user types
        placeholder="sk-...   Leave blank to skip AI explanations",
        label_visibility="collapsed"
    )
    st.caption(
        "Without a key you still get match scores and ranking. "
        "Add a key to also get 3-bullet GPT explanations per candidate. "
        "Get a free key at platform.openai.com."
    )

st.divider()

# =============================================================================
# MAIN BUTTON — runs when user clicks "Screen Resumes"
# =============================================================================

if st.button("Screen Resumes", type="primary", use_container_width=True):

    # --- Validation ---
    if not job_description.strip():
        st.error("Please paste a job description before screening.")
        st.stop()

    if not uploaded_files:
        st.error("Please upload at least one resume PDF.")
        st.stop()

    # --- Extract text from each PDF ---
    resumes = {}       # will hold {filename: raw_text}

    with st.spinner("Reading resumes..."):
        for f in uploaded_files:
            try:
                text = extract_text_from_pdf(f.read())
                if text:
                    resumes[f.name] = text
                else:
                    st.warning(f"Could not read text from {f.name} — it may be a scanned image PDF. Skipping.")
            except Exception as e:
                st.warning(f"Error reading {f.name}: {e}. Skipping.")

    if not resumes:
        st.error("No readable text found in any uploaded PDF. Make sure your PDFs are not scanned images.")
        st.stop()

    # --- Score all resumes ---
    with st.spinner("Calculating match scores..."):
        scores = score_resumes(job_description, resumes)

    # --- Sort by score, highest first ---
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # --- Show results ---
    st.success(f"Done. Screened {len(ranked)} candidates.")
    st.subheader("Candidate Rankings")

    for rank, (filename, score) in enumerate(ranked, start=1):

        # Make the display name cleaner — remove .pdf, replace underscores
        display_name = (
            filename
            .replace(".pdf", "")
            .replace("_", " ")
            .replace("-", " ")
            .title()
        )

        # Each candidate gets their own row with rank, name, score
        col_rank, col_name, col_score = st.columns([0.08, 0.67, 0.25])

        with col_rank:
            st.markdown(
                f"<div style='font-size:22px;font-weight:800;color:#4f9cf9;"
                f"padding-top:6px'>#{rank}</div>",
                unsafe_allow_html=True
            )

        with col_name:
            st.markdown(f"**{display_name}**")
            st.caption(filename)

        with col_score:
            # Color the score based on strength
            color = "#10b981" if score >= 60 else "#f59e0b" if score >= 35 else "#ef4444"
            st.markdown(
                f"<div style='font-size:22px;font-weight:700;color:{color}'>"
                f"{score}</div>"
                f"<div style='font-size:11px;color:#6b7280'>{score_label(score)}</div>",
                unsafe_allow_html=True
            )

        # Expandable section below each candidate
        with st.expander("See details"):
            if api_key:
                with st.spinner("Getting GPT explanation..."):
                    explanation = get_gpt_explanation(
                        job_description,
                        resumes[filename],
                        display_name,
                        score,
                        api_key
                    )
                if explanation:
                    st.write(explanation)
            else:
                # Show a preview of the resume text instead
                preview = resumes[filename][:600]
                if len(resumes[filename]) > 600:
                    preview += "..."
                st.text(preview)
                st.caption("Add an OpenAI API key above to get AI explanations here instead.")

        st.divider()

    # --- Summary at the bottom ---
    strong_count  = sum(1 for s in scores.values() if s >= 60)
    partial_count = sum(1 for s in scores.values() if 35 <= s < 60)
    weak_count    = sum(1 for s in scores.values() if s < 35)
    avg_score     = sum(scores.values()) / len(scores)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Screened",  len(ranked))
    col_b.metric("Strong Matches",  strong_count)
    col_c.metric("Partial Matches", partial_count)
    col_d.metric("Average Score",   f"{avg_score:.1f}")
