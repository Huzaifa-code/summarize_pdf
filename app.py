import streamlit as st
from transformers import pipeline
import pdfplumber

# Load the summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_chunk_length=512):
    # Break text into smaller chunks
    chunks = [text[i:i + max_chunk_length].strip() for i in range(0, len(text), max_chunk_length) if text[i:i + max_chunk_length].strip()]
    
    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    # Combine all summaries
    return " ".join(summaries)

# Streamlit app setup
st.title("Summarize PDF")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the PDF
    with pdfplumber.open(uploaded_file) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()
    
    # Summarize when button is clicked
    if st.button("Summarize"):
        if full_text:
            summary = summarize_text(full_text)
            st.write("Summary:")
            st.write(summary)
        else:
            st.write("No text found in the PDF.")
