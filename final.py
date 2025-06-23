import streamlit as st
import os
import tempfile
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import whisper
import ffmpeg
import pytesseract
import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(filename="lms_assistant.log", level=logging.ERROR)

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure Streamlit
st.set_page_config(page_title="Smart LMS Assistant", layout="centered")

@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("tiny")
    return whisper_model

# Preprocess image for Tesseract OCR
def preprocess_image(image):
    try:
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return Image.fromarray(img)
    except Exception as e:
        logging.error(f"Image preprocessing error: {str(e)}")
        st.error(f"Image preprocessing failed: {str(e)}")
        return image

# Gemini classification
def classify_with_gemini(text):
    try:
        prompt = (
            "You are evaluating student responses in a learning management system (LMS).\n"
            "Classify the following LMS text into one of these categories:\n"
            "- relevant: factually correct and contributes to the learning task\n"
            "- needs improvement: on-topic but vague, unclear, or underdeveloped\n"
            "- off-topic: unrelated to the subject matter\n"
            "- spam: promotional, copied, or inappropriate\n"
            "If the input is empty or too short, classify as 'needs improvement'.\n\n"
            f"Student response:\n{text}\n\n"
            "Respond with only the most suitable label."
        )
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, generation_config={"max_output_tokens": 100})
        return response.text.strip().lower()
    except Exception as e:
        logging.error(f"Classification error: {str(e)}")
        st.error(f"Classification failed: {str(e)}")
        return "error"

# Gemini enhancement
def enhance_with_gemini(text):
    try:
        prompt = (
            "Rewrite the following LMS content to be more detailed, educational, and contextually appropriate for learning.\n\n"
            f"Content:\n{text}\n\n"
            "Enhanced:"
        )
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, generation_config={"max_output_tokens": 500})
        return response.text.strip()
    except Exception as e:
        logging.error(f"Enhancement error: {str(e)}")
        st.error(f"Enhancement failed: {str(e)}")
        return text

# Gemini moderation
def moderate_gemini(text):
    try:
        prompt = (
            "Analyze this LMS content and identify any issues:\n"
            "- Is it offensive?\n"
            "- Is it spam or promotional?\n"
            "- Is it too short or low-effort?\n\n"
            f"Content:\n{text}\n\n"
            "List the issues found. If no issues, say 'No issues found.'"
        )
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, generation_config={"max_output_tokens": 250})
        return response.text.strip()
    except Exception as e:
        logging.error(f"Moderation error: {str(e)}")
        st.error(f"Moderation failed: {str(e)}")
        return "Moderation unavailable"

# Gemini Vision OCR with Tesseract fallback
def extract_text_with_gemini_vision(image):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            image,
            "Extract all readable handwritten or printed text from this image."
        ], generation_config={"max_output_tokens": 300})
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini Vision error: {str(e)}")
        st.warning(f"Gemini Vision failed: {str(e)}. Using Tesseract.")
        preprocessed_image = preprocess_image(image)
        return pytesseract.image_to_string(preprocessed_image)

# Whisper transcription
def transcribe_audio(path, whisper_model):
    try:
        result = whisper_model.transcribe(path)
        return result['text']
    except Exception as e:
        logging.error(f"Transcription error: {str(e)}")
        st.error(f"Transcription failed: {str(e)}")
        return ""

# Convert video to audio
def convert_video_to_audio(video_path, tmpdirname):
    audio_path = os.path.join(tmpdirname, "temp_audio.wav")
    try:
        ffmpeg.input(video_path).output(audio_path, ac=1, ar='16000').run(overwrite_output=True, quiet=True)
        return audio_path
    except ffmpeg.Error as e:
        logging.error(f"Video conversion error: {str(e)}")
        st.error(f"Video conversion failed: {str(e)}")
        return None

# Load model
whisper_model = load_models()

# UI
st.title("Smart LMS Assistant - Unified Multi-Modal Input")
input_type = st.radio("Choose input type:", ["Text", "Image", "Audio/Video"])
user_text = ""

# File size limit (15MB)
max_file_size = 15 * 1024 * 1024

if input_type == "Text":
    user_text = st.text_area("Enter LMS content:")

elif input_type == "Image":
    uploaded_image = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "png"])
    if uploaded_image:
        if uploaded_image.size > max_file_size:
            st.error("File size exceeds 15MB limit.")
        else:
            img = Image.open(uploaded_image)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            with st.spinner("Extracting text..."):
                user_text = extract_text_with_gemini_vision(img)

elif input_type == "Audio/Video":
    uploaded_file = st.file_uploader("Upload audio or video file", type=["mp3", "mp4", "wav", "mkv"])
    if uploaded_file:
        if uploaded_file.size > max_file_size:
            st.error("File size exceeds 15MB limit.")
        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_input_path = os.path.join(tmpdirname, uploaded_file.name)
                with open(temp_input_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                with st.spinner("Processing file..."):
                    if uploaded_file.type.startswith("video"):
                        audio_path = convert_video_to_audio(temp_input_path, tmpdirname)
                        if not audio_path:
                            st.stop()
                    else:
                        audio_path = temp_input_path

                    user_text = transcribe_audio(audio_path, whisper_model)

if user_text:
    st.subheader("Extracted Text")
    st.write(user_text)

    with st.spinner("Classifying..."):
        label = classify_with_gemini(user_text)
        st.subheader("Predicted Label")
        st.success(label)

    with st.spinner("Enhancing..."):
        enhanced = enhance_with_gemini(user_text.lower())
        st.subheader("Enhanced Version")
        st.info(enhanced)

    with st.spinner("Moderating..."):
        feedback = moderate_gemini(enhanced)
        st.subheader("Moderation Result")
        st.warning(feedback)