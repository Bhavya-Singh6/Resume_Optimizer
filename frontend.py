# app.py
import streamlit as st
from backend import run_backend

st.set_page_config(page_title="Tailor Job Applications", layout="wide")
st.title("🎯 Tailor Your Job Application with Multi-Agent AI (crewAI)")

# Inputs
job_desc = st.text_area("📄 Paste Job Description", height=200)
resume_text = st.text_area("👤 Paste Your Resume", height=200)

if st.button("Run Multi-Agent Crew"):
    if not job_desc or not resume_text:
        st.error("Please provide both Job Description and Resume.")
    else:
        with st.spinner("🤖 Agents are working..."):
            results = run_backend(job_desc, resume_text)

        st.success("✅ All agents finished their tasks!")

        # Timeline-like display
        with st.expander("🔎 Job Researcher Output"):
            st.markdown(results["job_analysis"])

        with st.expander("👤 Profile Analyzer Output"):
            st.markdown(results["profile_analysis"])

        with st.expander("📝 Resume Strategist Output (Tailored Resume)"):
            st.text_area("Tailored Resume", value=results["tailored_resume"], height=400)

        with st.expander("💬 Interview Coach Output (Prep Guide)"):
            st.text_area("Interview Prep", value=results["interview_prep"], height=300)

        # Download buttons
        st.download_button(
            "⬇️ Download Tailored Resume",
            results["tailored_resume"],
            file_name="tailored_resume.txt",
        )

        st.download_button(
            "⬇️ Download Interview Prep",
            results["interview_prep"],
            file_name="interview_prep.txt",
        )
