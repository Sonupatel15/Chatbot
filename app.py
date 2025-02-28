import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.title("📊 CDP Chatbot - Ask me anything!")

query = st.text_input("Ask a question about Segment, mParticle, Lytics, or Zeotap:")

if st.button("Get Answer"):
    if query:
        with st.spinner("Fetching answer..."):
            response = requests.post(API_URL, json={"question": query})
            answer = response.json().get("answer", "No answer found.")
            st.write("### Answer:")
            st.write(answer)
    else:
        st.warning("Please enter a question.")
