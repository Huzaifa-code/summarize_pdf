import streamlit as st
import pdfplumber
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file if running locally
load_dotenv()

# Retrieve the API key from the environment variable
API_KEY = os.getenv("HUGGING_FACE_API_KEY")
headers = {"Authorization": f"Bearer {API_KEY}"}

# Hugging Face API URL and headers
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

def query(payload):
    """Send text to Hugging Face API for summarization."""
    data = json.dumps(payload)
    response = requests.post(API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

def summarize_text(text, max_chunk_length=512):
    """Summarize text using Hugging Face API."""
    # Split text into smaller chunks for processing
    chunks = [text[i:i + max_chunk_length].strip() for i in range(0, len(text), max_chunk_length) if text[i:i + max_chunk_length].strip()]

    # Summarize each chunk using Hugging Face API
    summaries = []
    for chunk in chunks:
        response = query({
            "inputs": chunk,
            "parameters": {
                "max_length": 60,  # Adjust to shorten the summary
                "min_length": 20,  # Ensure summary contains enough info
            }
        })
        if 'error' in response:
            st.error(f"Error from Hugging Face API: {response['error']}")
            return "Error in summarization"
        summaries.append(response[0]['summary_text'])

    # Combine all summaries into one
    return " ".join(summaries)

# Streamlit app setup
st.title("Summarize PDF")


# Options for user input: either upload a PDF or type/paste text
input_option = st.radio("Choose input type:", ('Upload PDF', 'Enter Text'))


full_text = ""

if input_option == 'Upload PDF':
    # File uploader for PDF files
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        # Extract text from the uploaded PDF
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                full_text += page.extract_text()
        
        # Summarize when button is clicked
        if st.button("Summarize PDF"):
            if full_text:
                summary = summarize_text(full_text)
                st.write("Summary:")
                st.write(summary)
            else:
                st.write("No text found in the PDF.")
else:
    # Text input for direct text entry
    full_text = st.text_area("Enter text to summarize", height=200)
    # Summarize when button is clicked
    if st.button("Summarize Text"):
        if full_text.strip():
            summary = summarize_text(full_text)
            st.write("Summary:")
            st.write(summary)
        else:
            st.write("Please enter some text to summarize.")
