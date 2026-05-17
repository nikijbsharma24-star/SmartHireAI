import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Extract Text from PDF
# -----------------------------
def extract_text_from_pdf(uploaded_file):

    text = ""

    reader = PyPDF2.PdfReader(uploaded_file)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text


# -----------------------------
# Calculate Similarity
# -----------------------------
def calculate_similarity(resume_text, jd_text):

    documents = [resume_text, jd_text]

    tfidf = TfidfVectorizer()

    tfidf_matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    return similarity[0][0]


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AI Resume Shortlisting System")

st.write("Upload resumes and compare with Job Description")

# Job Description Input
jd_text = st.text_area(
    "Paste Job Description"
)

# Resume Upload
uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# Analyze Button
if st.button("Analyze Resumes"):

    if jd_text and uploaded_files:

        results = []

        for uploaded_file in uploaded_files:

            resume_text = extract_text_from_pdf(
                uploaded_file
            )

            score = calculate_similarity(
                resume_text,
                jd_text
            )

            results.append(
                (uploaded_file.name, score)
            )

        # Sort Results
        results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader("Ranking Results")

        for rank, (name, score) in enumerate(results, start=1):

            st.write(
                f"{rank}. {name} → Match Score: {round(score * 100, 2)}%"
            )

    else:

        st.warning("Please upload resumes and enter JD")
