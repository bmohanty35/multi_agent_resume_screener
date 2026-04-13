import streamlit as st
from backend import supervisor_agent

st.set_page_config(page_title="AI Resume Screener", layout="centered")

st.title("🤖 Multi-Agent Resume Screener")

resume = st.text_area("Paste Resume")
job_desc = st.text_area("Paste Job Description")

if st.button("Analyze"):
    if resume and job_desc:
        with st.spinner("Analyzing..."):
            result = supervisor_agent.invoke({
                "input": f"Resume: {resume} Job: {job_desc}"
            })

            st.success("Analysis Complete ✅")
            st.markdown(result["messages"][-1].content)
    else:
        st.warning("Please fill both fields")