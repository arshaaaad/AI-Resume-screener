# AI Resume Screener

Upload a job description and multiple resumes. The app scores every candidate 0 to 100
based on how well their resume matches the job, ranks them, and optionally uses GPT to
explain in 3 bullet points why each person fits or does not.

Built with Python, scikit-learn (TF-IDF + cosine similarity), and Streamlit.


## Live Demo

Try it here:(https://ai-resume-screener-rzbwxrrnh6v5dr5tfspcdt.streamlit.app/)



## The problem it solves

A recruiter posts a job and receives 50 resumes. Reading all of them takes 3 hours.
Most are completely irrelevant.

This app reads all 50 in 10 seconds, scores each one, and ranks them so the recruiter
only needs to read the top few.



## How it works

Step 1 — Extract text from each uploaded PDF using PyPDF2.

Step 2 — Clean the text by removing URLs, symbols, and punctuation.

Step 3 — Convert all documents (job description + all resumes) into TF-IDF vectors
using scikit-learn. TF-IDF weights rare technical keywords like PyTorch or Kubernetes
higher than common words like experience or team.

Step 4 — Calculate cosine similarity between the job description vector and each
resume vector. Cosine similarity measures the angle between two vectors — small angle
means similar content. Multiply by 100 to get a 0-100 score.

Step 5 — Rank candidates from highest to lowest score.

Step 6 (optional) — Send the JD and resume to GPT-3.5 to get 3 bullet points
explaining why each candidate fits or does not.



## Tech stack

- scikit-learn — TF-IDF vectorizer and cosine similarity
- PyPDF2 — extract text from PDF resumes
- OpenAI API — GPT-3.5 explanations per candidate
- Streamlit — web UI
- Streamlit Cloud — free deployment


## How to run locally

    git clone https://github.com/YOUR_USERNAME/ai-resume-screener
    cd ai-resume-screener
    pip install -r requirements.txt
    streamlit run app.py

Open http://localhost:8501 in your browser.

No model files needed. Everything is computed fresh each time.



## Project structure

    ai-resume-screener/
        app.py                      main Streamlit application
        requirements.txt            dependencies
        README.md                   this file



## Resume bullet point

Built AI resume screening system using TF-IDF cosine similarity and GPT-3.5 to rank
candidates and explain fit; reduces manual screening time by around 80%. 



## Why cosine similarity and not keyword counting

Keyword counting treats all words equally and rewards longer resumes.
Cosine similarity is length-independent — a one-page and three-page resume with the
same skills score the same. TF-IDF also penalises generic words and rewards rare
technical terms, which is exactly what matters for job matching.
