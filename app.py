import streamlit as st 
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai
import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os

# Load API Keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


# Page configuration
st.set_page_config(
    page_title="Multimodal AI Agent - Video & Image Analyzer",
    page_icon="🎥📸",
    layout="wide"
)

st.title("Phidata AI Agent 🎥📸 - Video & Image Analyzer")

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Multimodal AI Analyzer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

## Initialize the agent
multimodal_Agent = initialize_agent()

# File uploaders for video and images
video_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'], help="Upload a video for AI analysis")
image_files = st.file_uploader("Upload one or more images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, help="Upload images for AI analysis")

video_path, image_paths = None, []

# Handle video upload
if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name
    st.video(video_path, format="video/mp4", start_time=0)

# Display uploaded images
if image_files:
    st.subheader("Uploaded Images")
    for img_file in image_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img:
            temp_img.write(img_file.read())
            image_paths.append(temp_img.name)
        st.image(temp_img.name, caption=img_file.name, use_column_width=True)

# User query input
user_query = st.text_area(
    "What insights are you seeking?",
    placeholder="Ask anything about the video and images. The AI agent will analyze and gather additional context if needed.",
    help="Provide specific questions or insights you want from the media.",
)

if st.button("🔍 Analyze Media", key="analyze_media_button"):
    if not user_query:
        st.warning("Please enter a question or insight to analyze the media.")
    else:
        try:
            with st.spinner("Processing media and gathering insights..."):
                uploaded_files = []
                
                # Upload and process video file
                if video_path:
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)
                    uploaded_files.append(processed_video)
                
                # Upload and process images
                processed_images = []
                for img_path in image_paths:
                    processed_img = upload_file(img_path)
                    processed_images.append(processed_img)

                # Construct analysis prompt
                analysis_prompt = f"""
                Analyze the uploaded media (video and images) for content and context.
                Respond to the following query using insights from the provided media and supplementary web research:
                {user_query}

                Media Details:
                - Number of Images: {len(processed_images)}
                - Video Uploaded: {'Yes' if video_path else 'No'}

                Provide a detailed, user-friendly, and actionable response.
                """

                # AI agent processing
                response = multimodal_Agent.run(
                    analysis_prompt, 
                    videos=uploaded_files, 
                    images=processed_images
                )

            # Display the result
            st.subheader("Analysis Result")
            st.markdown(response.content)

        except Exception as error:
            st.error(f"An error occurred during analysis: {error}")
        finally:
            # Clean up temporary files
            if video_path:
                Path(video_path).unlink(missing_ok=True)
            for img_path in image_paths:
                Path(img_path).unlink(missing_ok=True)

# Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
