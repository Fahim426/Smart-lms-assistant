# Multi-Modal AI Assistant for LMS Content Moderation and Enhancement

## Project Overview
This project delivers a Smart LMS Assistant — an AI-powered solution designed to analyze, enhance, and moderate multi-modal educational content within a Learning Management System (LMS). Built as part of an internship project, it intelligently supports both educators and students by improving the quality, relevance, and clarity of LMS submissions across text, image, and audio/video formats.

Key functionalities include:
 Classification of LMS responses using Gemini (relevant, off-topic, spam, needs improvement)
 Generative enhancement of vague or unclear content
 Moderation checks for spam, offensive, or low-effort content
 OCR using Gemini Vision with fallback to Tesseract (with preprocessing)
 Audio/Video transcription using Whisper + FFmpeg
 Streamlit web interface with upload support and real-time AI feedback

## Project Structure
ai_assistant_project/
├── .env
├── Dockerfile
├── docker-compose.yml
├── final.py
└── README.md            # Project documentation
├── .gitignore

# Build and run the app using Docker Compose
docker-compose up --build
