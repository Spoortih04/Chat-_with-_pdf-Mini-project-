import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
import fitz  # PyMuPDF library

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="CHAT WITH PDF...!",
    page_icon=":crop:",  # Favicon emoji
    layout="centered",  # Page layout option
)

# Load API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if the API key is set
if not GOOGLE_API_KEY:
    st.error("Google API key is not set. Please check your environment variables.")
else:
    # Set up Google Gemini-Pro AI model
    gen_ai.configure(api_key=GOOGLE_API_KEY)
    model = gen_ai.GenerativeModel('gemini-pro')

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_file):
    try:
        # Open the PDF file as a file-like object
        pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
        text = ""
        for page in pdf_document:
            # Extract text from the page
            text += page.get_text()
        pdf_document.close()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    try:
        st.session_state.chat_session = model.start_chat(history=[])
    except Exception as e:
        st.error(f"Failed to start chat session: {e}")

# Display the chatbot's title on the page
st.title("🤖 CHAT WITH PDF 📚")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
pdf_text = ""

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file.read())
    if pdf_text is None:
        st.warning("Failed to extract text from PDF. Please upload a valid PDF file.")

# Display the chat history
if "chat_session" in st.session_state:
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

# Input field for user's message
user_prompt = st.chat_input("Ask CHAT WITH PDF...")
if user_prompt:
    if "chat_session" not in st.session_state:
        st.error("Chat session is not initialized.")
    else:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        if pdf_text:
            # Combine the user's question with the extracted PDF text
            combined_prompt = f"Extracted PDF Text: {pdf_text[:1000]}... \n\nQuestion: {user_prompt}"
        else:
            combined_prompt = f"Question: {user_prompt}"

        try:
            # Send user's message and PDF context to Gemini-Pro and get the response
            gemini_response = st.session_state.chat_session.send_message(combined_prompt)

            # Display Gemini-Pro's response
            with st.chat_message("assistant"):
                st.markdown(gemini_response.text)
        except Exception as e:
            st.error(f"Error communicating with the AI model: {e}")