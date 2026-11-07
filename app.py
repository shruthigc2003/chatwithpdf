import os
import re
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PREFERRED_MODEL = "gemini-2.5-flash"

st.set_page_config(page_title="ChatWithPDF", page_icon="📄")
st.title("📄 ChatWithPDF: Ask Your Notes Anything")


@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = get_model()


def clean(text):
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)


def load_chunks(file, size=500, overlap=50):
    pages = []
    for p in PdfReader(file).pages:
        try:
            pages.append(clean(p.extract_text() or ""))
        except Exception:
            continue
    words = " ".join(pages).split()
    chunks = [" ".join(words[i:i + size]) for i in range(0, len(words), size - overlap)]
    return [str(c) for c in chunks if c.strip()]


def top_chunks(question, chunks, embeds, k=4):
    q = model.encode([clean(question)])[0]
    scores = embeds @ q / (np.linalg.norm(embeds, axis=1) * np.linalg.norm(q))
    return [chunks[i] for i in scores.argsort()[-k:][::-1]]


def ask_llm(question, context):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    from google import genai

    client = genai.Client(api_key=api_key)
    prompt = (
        "Answer only using the context below. "
        "If the answer is not there, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    def generate(model_name):
        return client.models.generate_content(model=model_name, contents=prompt).text

    try:
        return generate(PREFERRED_MODEL)
    except Exception as first_error:
        # fall back to any other Flash model that this key can use
        names = [
            m.name.replace("models/", "")
            for m in client.models.list()
            if "generateContent" in (m.supported_actions or [])
        ]
        skip = ("image", "tts", "live", "audio", "thinking", "embedding")
        candidates = [n for n in names if "flash" in n and not any(s in n for s in skip)]
        for name in candidates:
            if name == PREFERRED_MODEL:
                continue
            try:
                return generate(name)
            except Exception:
                continue
        raise first_error


uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    if st.session_state.get("file_name") != uploaded.name:
        with st.spinner("Reading your PDF... this can take a minute for big files"):
            chunks = load_chunks(uploaded)
            if not chunks:
                st.error("No text found. This may be a scanned PDF.")
                st.stop()
            st.session_state.chunks = chunks
            st.session_state.embeds = model.encode(chunks, show_progress_bar=False)
            st.session_state.file_name = uploaded.name

    question = st.text_input("Ask a question about your PDF")
    if question:
        found = top_chunks(question, st.session_state.chunks, st.session_state.embeds)
        try:
            answer = ask_llm(question, "\n\n".join(found))
        except Exception as e:
            answer = None
            st.error(f"LLM error: {e}")
        if answer:
            st.subheader("Answer")
            st.write(answer)
        else:
            st.info("No answer from the LLM, so showing the most relevant passages only.")
        with st.expander("Sources from your PDF"):
            for c in found:
                st.write(c)
                st.divider()