import pytest
from unittest.mock import patch, MagicMock

# Adjust the import path based on your project structure if necessary
from the_bot.agents.tools.summarize_text import summarize_text

@patch('the_bot.agents.tools.summarize_text.current_agent')
def test_summarize_text_tool(mock_current_agent):
    # Arrange
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "This is a test summary."

    mock_agent_instance = MagicMock()
    mock_agent_instance.model = mock_llm

    mock_current_agent.return_value = mock_agent_instance

    sample_text = "This is a long piece of text that needs to be summarized. It talks about many things and has several important points."
    max_len = 50

    expected_prompt = f"Summarize the following in <={max_len} words:\n\n{sample_text}"

    # Act
    # The .run() method is used for Langchain tools when they are invoked by an agent.
    # If summarize_text is a regular function decorated with @tool,
    # and you're testing its direct callable form, you might call it as summarize_text(text=sample_text, max_length=max_len)
    # However, the provided math tests used .run(args_dict), so sticking to that for consistency with @tool
    summary_result = summarize_text.run({"text": sample_text, "max_length": max_len})

    # Assert
    mock_current_agent.assert_called_once()
    mock_agent_instance.model.generate.assert_called_once_with(expected_prompt)
    assert summary_result == "This is a test summary."

@patch('the_bot.agents.tools.summarize_text.current_agent')
def test_summarize_text_tool_default_max_length(mock_current_agent):
    # Arrange
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Default length summary."

    mock_agent_instance = MagicMock()
    mock_agent_instance.model = mock_llm

    mock_current_agent.return_value = mock_agent_instance

    sample_text = "Another text to summarize."
    default_max_len = 200  # Default value from the function signature

    expected_prompt = f"Summarize the following in <={default_max_len} words:\n\n{sample_text}"

    # Act
    summary_result = summarize_text.run({"text": sample_text}) # max_length not provided, should use default

    # Assert
    mock_current_agent.assert_called_once()
    mock_agent_instance.model.generate.assert_called_once_with(expected_prompt)
    assert summary_result == "Default length summary."
