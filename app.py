"""
Resume Optimizer - Streamlit Frontend
Multi-agent AI application for tailoring resumes and interview preparation.
"""
import streamlit as st
from backend import run_backend
from PyPDF2 import PdfReader
from utils import extract_text_from_pdf
from typing import Dict, Any

# -------------------------------
# Constants
# -------------------------------
DEFAULT_TEXT_AREA_HEIGHT = 200
RESUME_TEXT_AREA_HEIGHT = 400
INTERVIEW_TEXT_AREA_HEIGHT = 300

# -------------------------------
# Helper Functions
# -------------------------------
def extract_pdf_text(resume_file) -> str:
    """
    Extract text from uploaded PDF file.
    
    Args:
        resume_file: Streamlit uploaded file object
        
    Returns:
        Extracted text from PDF
    """
    resume_text = ""
    try:
        pdf_reader = PdfReader(resume_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                resume_text += page_text + "\n"
    except Exception:
        # Fallback using utility (handles file-like objects by saving temp file)
        resume_text = extract_text_from_pdf(resume_file)
    
    return resume_text


def display_results(results: Dict[str, Any]) -> None:
    """
    Display agent outputs in expandable sections.
    
    Args:
        results: Dictionary containing agent outputs
    """
    st.success("✅ All agents finished their tasks!")
    st.header("📊 Agent Outputs")
    
    # Define sections to display
    sections = [
        ("🔎 Job Researcher Output", "job_analysis", False),
        ("👤 Profile Analyzer Output", "profile_analysis", False),
        ("📝 Resume Strategist Output (Tailored Resume)", "tailored_resume", True),
        ("💬 Interview Coach Output (Prep Guide)", "interview_prep", True)
    ]
    
    for title, key, use_text_area in sections:
        with st.expander(title):
            content = results.get(key, "No output.")
            if use_text_area:
                height = RESUME_TEXT_AREA_HEIGHT if "Resume" in title else INTERVIEW_TEXT_AREA_HEIGHT
                st.text_area(key.replace("_", " ").title(), value=content, height=height, key=f"{key}_display")
            else:
                st.markdown(content)


def display_download_buttons(results: Dict[str, Any]) -> None:
    """
    Display download buttons for tailored resume and interview prep.
    
    Args:
        results: Dictionary containing agent outputs
    """
    st.header("⬇️ Download Results")
    
    if results.get("tailored_resume"):
        st.download_button(
            "Download Tailored Resume",
            results["tailored_resume"],
            file_name="tailored_resume.txt",
            mime="text/plain"
        )
    
    if results.get("interview_prep"):
        st.download_button(
            "Download Interview Prep",
            results["interview_prep"],
            file_name="interview_prep.txt",
            mime="text/plain"
        )


# -------------------------------
# Streamlit App Configuration
# -------------------------------
st.set_page_config(
    page_title="Tailor Job Applications",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Tailor Your Job Application with Multi-Agent AI (CrewAI)")

st.markdown("""
This app helps you **tailor your resume** and **prepare for interviews** using
a team of AI agents:

- 🔎 **Job Researcher** → Analyzes job description
- 👤 **Profile Analyzer** → Reads your resume PDF
- 📝 **Resume Strategist** → Creates tailored resume
- 💬 **Interview Coach** → Suggests interview questions & answers
""")

# -------------------------------
# Input Section
# -------------------------------
st.header("📥 Input Data")

job_desc = st.text_area(
    "Paste Job Description",
    height=200,
    placeholder="Paste the full job description here..."
)

resume_file = st.file_uploader(
    "Upload Your Resume (PDF)",
    type=["pdf"]
)

# -------------------------------
# Run Button
# -------------------------------
if st.button("🚀 Run Multi-Agent Crew"):
    if not job_desc:
        st.error("❌ Please provide Job Description.")
    elif not resume_file:
        st.error("❌ Please upload your Resume PDF.")
    else:
        # Extract text from PDF
        resume_text = extract_pdf_text(resume_file)

        if not resume_text.strip():
            st.error("❌ Unable to extract text from PDF. Make sure it is readable (not a scanned image).")
        else:
            # Run backend with spinner
            with st.spinner("🤖 Agents are working together..."):
                results = run_backend(job_desc, resume_text)

            # Check for errors
            if results.get("error"):
                st.error(f"Error: {results['error']}")
            else:
                display_results(results)
                display_download_buttons(results)
