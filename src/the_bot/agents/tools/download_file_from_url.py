import os
import tempfile
import requests
from urllib.parse import urlparse

from langchain_core.tools import tool

@tool
def download_file_from_url(url: str, filename: str | None = None) -> str:
    """
    Download a file from a URL and save it to a temporary location.

    Args:
    -----
    url : str
        The URL from which the file will be downloaded. This must be a valid URL pointing
        to the file you want to retrieve.

    filename : str, optional
        The name of the file to save the downloaded content as. If not provided, the
        filename is derived from the URL. If the URL does not include a valid filename,
        a random filename will be generated. The default is `None`.

    Returns:
    --------
    str
        The full file path where the downloaded file is saved in the temporary directory.

    """
    # Determine filename
    if not filename:
        path = urlparse(url).path
        filename = os.path.basename(path) or f"download_{os.urandom(4).hex()}"
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(filepath, 'wb') as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    return filepath
