from langchain_core.tools import tool

@tool
def summarize_text(text: str, max_length: int = 200) -> str:
    """
    Summarize text using the agent's LLM model.

    Args:
    -----
    text : str
        The text to be summarized. This is the required input, and the function will
        generate a summary based on its content.

    max_length : int, optional
        The maximum length (in words) of the generated summary. The default is 200 words.
        The summary will be truncated to this length, or less if the content can be summarized
        more concisely.

    Returns:
    --------
    str
        A summarized version of the input text, with a length that does not exceed
        the specified `max_length`.
    """
    # Assume the agent context or direct LLM invocation is available
    # This example uses a placeholder invocation:
    from smolagents import current_agent
    prompt = f"Summarize the following in <={max_length} words:\n\n{text}"
    return current_agent().model.generate(prompt)
