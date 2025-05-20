import base64
import mimetypes
import os
from urllib.parse import urlparse

import yt_dlp
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm(model_id: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return ChatGoogleGenerativeAI(
        model=model_id,
        temperature=0,
        max_retries=2,
        google_api_key=api_key
    )


def encode_file(file_path: str) -> (str, str):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        raise RuntimeError(f"Could not determine MIME type for {file_path}")
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded, mime_type


@tool
def image_analysis_tool(question: str, file_path: str, model_id: str = "gemini-2.5-flash-preview-04-17") -> str:
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
        llm = get_llm(model_id)
        encoded_image, mime_type = encode_file(file_path)

        message = HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": f"data:{mime_type};base64,{encoded_image}"}
        ])

        result = llm.invoke([message])
        return result.content
    except Exception as e:
        raise RuntimeError(f"Processing failed: {e}") from e


@tool
def audio_analysis_tool(question: str, file_path: str, model_id: str = "gemini-2.5-flash-preview-04-17") -> str:
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
        llm = get_llm(model_id)
        encoded_audio, mime_type = encode_file(file_path)

        message = HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "media", "mime_type": mime_type, "data": encoded_audio}
        ])

        result = llm.invoke([message])
        return result.content
    except Exception as e:
        raise RuntimeError(f"Processing failed: {e}") from e


@tool
def video_analysis_tool(question: str, file_path: str, model_id: str = "gemini-2.5-flash-preview-04-17") -> str:
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
        llm = get_llm(model_id)
        encoded_video, mime_type = encode_file(file_path)

        message = HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "media", "mime_type": mime_type, "data": encoded_video}
        ])

        result = llm.invoke([message])
        return result.content
    except Exception as e:
        raise RuntimeError(f"Processing failed: {e}") from e


@tool
def youtube_analysis_tool(question: str, url: str, model_id: str = "gemini-2.5-flash-preview-04-17") -> str:
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
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return "Please provide a valid video URL with http:// or https:// prefix."

        if 'youtube.com' not in url and 'youtu.be' not in url:
            return "Only YouTube videos are supported."

        llm = get_llm(model_id)

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'no_playlist': True,
            'youtube_include_dash_manifest': False
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False, process=False)
            if not info:
                return "Could not extract video information."

            title = info.get('title', 'Unknown')
            description = info.get('description', 'No description provided.')

            prompt = (
                f"Analyze this YouTube video metadata:\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Description: {description}\n"
                f"Question: {question}\n"
                f"Focus your answer on:\n"
                f"1. Main topic and key points\n"
                f"2. Expected visual elements\n"
                f"3. Overall message or purpose\n"
                f"4. Target audience"
            )

            messages = [HumanMessage(content=prompt)]
            response = llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)

    except Exception as e:
        raise RuntimeError(f"Error analyzing video: {e}") from e


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
