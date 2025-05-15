import os
import tempfile

from langchain_core.tools import tool

@tool
def save_and_read_file(content: str, filename: str | None = None) -> str:
    """
    Save content to a temporary file and return the path.

    Args:
    -----
    content : str
        The content to be written to the file. This is a required argument and must be
        a string containing the data to be saved.

    filename : str, optional
        The name of the file where the content will be saved. If not provided, a temporary
        file will be created with a randomly generated name. The default is `None`.

    Returns:
    --------
    str
        The file path where the content was saved. This can be the path to the provided
        `filename` or a path to a randomly generated temporary file if `filename` was not given.
    """
    temp_dir = tempfile.gettempdir()
    if filename:
        path = os.path.join(temp_dir, filename)
    else:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        path = temp_file.name
    with open(path, 'w') as f:
        f.write(content)
    return path
