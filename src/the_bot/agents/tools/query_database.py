# import json

# from langchain_core.tools import tool

# @tool
# def query_database(conn_str: str, query: str) -> str:
#     """
#     Connect to a SQL DB and run a query. Returns JSON-encoded results.
#     """
#     import sqlalchemy
#     engine = sqlalchemy.create_engine(conn_str)
#     with engine.connect() as conn:
#         result = conn.execute(sqlalchemy.text(query))
#         rows = [dict(r) for r in result]
#     return json.dumps(rows)


# # File: src/the_bot/agents/tools/summarize_text.py
# from smolagents import tool

# @tool
# def summarize_text(text: str, max_length: int = 200) -> str:
#     """
#     Summarize text using the agent's LLM model.
#     """
#     # Assume the agent context or direct LLM invocation is available
#     # This example uses a placeholder invocation:
#     from smolagents import current_agent
#     prompt = f"Summarize the following in <={max_length} words:\n\n{text}"
#     return current_agent().model.generate(prompt)


# # File: src/the_bot/agents/tools/save_and_read_file.py
# from smolagents import tool
# import os
# import tempfile

# @tool
# def save_and_read_file(content: str, filename: str = None) -> str:
#     """
#     Save content to a temporary file and return the path.
#     """
#     temp_dir = tempfile.gettempdir()
#     if filename:
#         path = os.path.join(temp_dir, filename)
#     else:
#         temp_file = tempfile.NamedTemporaryFile(delete=False)
#         path = temp_file.name
#     with open(path, 'w') as f:
#         f.write(content)
#     return path


# # File: src/the_bot/agents/tools/download_file_from_url.py
# from smolagents import tool
# import os
# import tempfile
# import requests
# from urllib.parse import urlparse

# @tool
# def download_file_from_url(url: str, filename: str = None) -> str:
#     """
#     Download a file from a URL and save it to a temporary location.
#     """
#     # Determine filename
#     if not filename:
#         path = urlparse(url).path
#         filename = os.path.basename(path) or f"download_{os.urandom(4).hex()}"
#     temp_dir = tempfile.gettempdir()
#     filepath = os.path.join(temp_dir, filename)
#     resp = requests.get(url, stream=True)
#     resp.raise_for_status()
#     with open(filepath, 'wb') as f:
#         for chunk in resp.iter_content(8192):
#             f.write(chunk)
#     return filepath


# # File: src/the_bot/agents/tools/extract_text_from_image.py
# from smolagents import tool

# @tool
# def extract_text_from_image(image_path: str) -> str:
#     """
#     Extract text from an image using pytesseract.
#     """
#     try:
#         from PIL import Image
#         import pytesseract
#     except ImportError:
#         return "Error: install Pillow and pytesseract."
#     img = Image.open(image_path)
#     return pytesseract.image_to_string(img)


# # File: src/the_bot/agents/tools/analyze_csv_file.py
# from smolagents import tool

# @tool
# def analyze_csv_file(file_path: str, query: str = None) -> str:
#     """
#     Load a CSV and return summary statistics.
#     """
#     import pandas as pd
#     df = pd.read_csv(file_path)
#     summary = df.describe(include='all').to_string()
#     return f"Rows: {len(df)}, Columns: {len(df.columns)}\n
# Cols: {', '.join(df.columns)}\n
# Summary:\n{summary}"


# # File: src/the_bot/agents/tools/analyze_excel_file.py
# from smolagents import tool

# @tool
# def analyze_excel_file(file_path: str, query: str = None) -> str:
#     """
#     Load an Excel file and return summary statistics.
#     """
#     import pandas as pd
#     df = pd.read_excel(file_path)
#     summary = df.describe(include='all').to_string()
#     return f"Rows: {len(df)}, Columns: {len(df.columns)}\n
# Cols: {', '.join(df.columns)}\n
# Summary:\n{summary}"
