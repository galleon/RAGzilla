from langchain_core.tools import tool


@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using pytesseract.

    Args:
    -----
    image_path: str
        The file path to the image from which text will be extracted. This should be
        the path to a valid image file that can be processed by the `PIL.Image.open` function.

    Returns:
    --------
    str
        The text extracted from the image. If an error occurs during the import of
        the necessary libraries, an error message is returned.

    Raises:
    -------
    ImportError
        If the required libraries `Pillow` and `pytesseract` are not installed, the function
        will return an error message instructing the user to install them.image_path : str
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return "Error: install Pillow and pytesseract."
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)
