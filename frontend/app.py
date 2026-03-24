import streamlit as st
import requests

import re

BACKEND_URL = "http://127.0.0.1:8001"

st.title("Talk with your PDF RAG System")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/pdf"
        )
    }

    response = requests.post(
        f"{BACKEND_URL}/upload",
        files=files
    )
    st.write(response.text)
question = st.text_input("Ask Question")

if st.button("Ask"):
    response = requests.get(
        f"{BACKEND_URL}/ask",
        params={"question": question}
    )
    answer = response.json()["answer"]

    st.markdown("## Answer")

    # detect LaTeX math
    latex_blocks = re.findall(
        r'\$\$(.*?)\$\$|\\\[(.*?)\\\]',
        answer,
        re.DOTALL
    )

    if latex_blocks:

        st.markdown(answer)

        for block in latex_blocks:

            formula = block[0] or block[1]

            st.latex(formula)

    else:

        st.markdown(answer)
