# 📄 ChatWithPDF

Ask questions about any PDF in plain English. The app finds the most relevant parts of the document and uses an LLM to answer using only that content.

## How it works
1. The PDF is read and split into overlapping chunks of text.
2. Each chunk is converted into an embedding with `sentence-transformers`.
3. Your question is embedded and compared with the chunks using cosine similarity.
4. The top matching chunks are sent to the Gemini API, which answers using only that context.
5. The source passages are shown below the answer, so you can verify it.

## Tech stack
Python, Streamlit, pypdf, sentence-transformers (all-MiniLM-L6-v2), NumPy, Google Gemini API

## Run it locally
```bash
git clone https://github.com/shruthigc2003/chatwithpdf.git
cd chatwithpdf
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Set your Gemini API key (get a free one at aistudio.google.com):
```powershell
$env:GEMINI_API_KEY="your_key_here"
```
Then run:
```bash
streamlit run app.py
```

## Features
- Works with text-based PDFs
- Answers only from the document, and says "I don't know" when the answer isn't there
- Shows the source passages for every answer
- Skips unreadable pages instead of crashing

## What I'd improve
- Support multiple PDFs
- Chat history for follow-up questions
- OCR for scanned PDFs
- A vector database for very large documents

## Author
Shruthi, MSc Computer Science student, interested in AI
