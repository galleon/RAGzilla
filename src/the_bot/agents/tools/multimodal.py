import base64
import os
from urllib.parse import urlparse


import yt_dlp
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

@tool
def image_analysis_tool(
    question: str,
    file_path: str,
    model_id: str = "gemini-2.5-flash-preview-04-17"
) -> str:
    """
    Given a question and image file, analyze the image to answer the question.

    Args:
        question (str): Question about an image file
        file_path (str): The image file path

    Returns:
        str: Answer to the question about the image file

    Raises:
        RuntimeError: If processing fails
    """
    try:
        google_api_key = os.getenv("GEMINI_API_KEY", None)

        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0,
            max_retries=2,
            google_api_key=google_api_key
        )

        with open(file_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": f"data:image/png;base64,{encoded_image}"},
            ]
        )

        result = llm.invoke([message])

        return result.content
    except Exception as e:
        raise RuntimeError(f"Processing failed: {str(e)}")

@tool
def audio_analysis_tool(
    question: str,
    file_path: str,
    model_id: str = "gemini-2.5-flash-preview-04-17"
) -> str:
    """
    Given a question and local audio file, analyze the audio to answer the question.

    Args:
        question (str): Question about an audio file
        file_path (str): The audio file path

    Returns:
        str: Answer to the question about the audio file

    Raises:
        RuntimeError: If processing fails
    """
    try:
        google_api_key = os.getenv("GEMINI_API_KEY", None)

        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0,
            max_retries=2,
            google_api_key=google_api_key,
        )

        with open(file_path, "rb") as image_file:
            encoded_audio = base64.b64encode(image_file.read()).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                {
                    "type": "media",
                    "mime_type": "audio/mpeg",
                    "data": f"{encoded_audio}"
                },
            ]
        )

        result = llm.invoke([message])

        return result.content
    except Exception as e:
        raise RuntimeError(f"Processing failed: {str(e)}")

@tool
def video_analysis_tool(
    question: str,
    file_path: str,
    model_id: str = "gemini-2.5-flash-preview-04-17"
) -> str:
    """
    Given a question and a local video file, analyze the video to answer the question.

    Args:
        question (str): Question about a video file
        file_path (str): The video file path

    Returns:
        str: Answer to the question about the video file

    Raises:
        RuntimeError: If processing fails
    """
    try:
        google_api_key = os.getenv("GEMINI_API_KEY", None)

        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0,
            max_retries=2,
            google_api_key=google_api_key
        )

        with open(file_path, "rb") as image_file:
            encoded_video = base64.b64encode(image_file.read()).decode("utf-8")

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": question
                },
                {
                    "type": "media",
                    "mime_type": "video/mp4",
                    "image_url": f"{encoded_video}"
                },
            ]
        )

        result = llm.invoke([message])

        return result.content
    except Exception as e:
        raise RuntimeError(f"Processing failed: {str(e)}")

@tool
def youtube_analysis_tool(
    question: str,
    url: str,
    model_id: str = "gemini-2.5-flash-preview-04-17"
) -> str:
    """
    Given a question and YouTube URL, analyze the video to answer the question.

    Args:
        question (str): Question about a YouTube video
        url (str): The YouTube video URL

    Returns:
        str: Answer to the question about the YouTube video

    Raises:
           RuntimeError: If processing fails
    """
    try:
        google_api_key = os.getenv("GEMINI_API_KEY", None)

        # Validate URL
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            return "Please provide a valid video URL with http:// or https:// prefix."

        # Check if it's a YouTube URL
        if 'youtube.com' not in url and 'youtu.be' not in url:
            return "Only YouTube videos are supported at this time."


        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0,
            max_retries=2,
            google_api_key=google_api_key
        )

        try:
            # Configure yt-dlp with minimal extraction
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'no_playlist': True,
                'youtube_include_dash_manifest': False
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    # Try basic info extraction
                    info = ydl.extract_info(url, download=False, process=False)
                    if not info:
                        return "Could not extract video information."

                    title = info.get('title', 'Unknown')
                    description = info.get('description', '')

                    # Create a detailed prompt with available metadata
                    prompt = f"""Please analyze this YouTube video:
Title: {title}
URL: {url}
Description: {description}
Question: {question}
Please answer the question focusing on:
1. Main topic and key points from the title and description
2. Expected visual elements and scenes
3. Overall message or purpose
4. Target audience"""

                    print(prompt)
                    # Use the LLM with proper message format
                    messages = [HumanMessage(content=prompt)]
                    response = llm.invoke(messages)
                    print(type(response))
                    return response.content if hasattr(response, 'content') else str(response)

                except Exception as e:
                    if 'Sign in to confirm' in str(e):
                        return "This video requires age verification or sign-in. Please provide a different video URL."
                    return f"Error accessing video: {str(e)}"

        except Exception as e:
            return f"Error extracting video info: {str(e)}"

    except Exception as e:
        return f"Error analyzing video: {str(e)}"

# @tool
# def document_analysis_tool(
#     question: str,
#     file_path: str,
#     model_id: str = "gemini-2.5-flash-preview-04-17"
# ) -> str:
#     """
#     Given a question and document file, analyze the document to answer the question.

#     Args:
#         question (str): Question about a document file
#         file_path (str): The document file path

#     Returns:
#         str: Answer to the question about the document file

#     Raises:
#         RuntimeError: If processing fails
#     """
#     print(__file__)

#     try:
#         client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

#         contents = []

#         ext = os.path.splitext(file_path)[1].lower()

#         if ".docx" == ext:
#             text_data = read_docx_text(file_path)
#             contents = [f"{question}\n{text_data}"]
#             print(f"=> Text data:\n{text_data}")
#         elif ".pptx" == ext:
#             text_data = read_pptx_text(file_path)
#             contents = [f"{question}\n{text_data}"]
#             print(f"=> Text data:\n{text_data}")
#         else:
#             file = client.files.upload(file=file_path)
#             contents = [file, question]

#         response = client.models.generate_content(
#             model=model_id,
#             contents=contents
#         )

#         return response.text
#     except Exception as e:
#         raise RuntimeError(f"Processing failed: {str(e)}")
