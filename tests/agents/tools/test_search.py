import pytest
from unittest.mock import patch, MagicMock, ANY
from langchain_core.documents import Document # For creating mock Document objects

# Adjust the import path based on your project structure if necessary
from the_bot.agents.tools.search import wiki_search, web_search, arxiv_search

# --- Wikipedia Search Tests ---
@patch('the_bot.agents.tools.search.WikipediaLoader')
def test_wiki_search_success(MockWikipediaLoader):
    # Arrange
    mock_doc1 = Document(page_content="Content of page 1", metadata={"source": "http://wiki.com/page1", "page": "Page 1 Title"})
    mock_doc2 = Document(page_content="Content of page 2", metadata={"source": "http://wiki.com/page2", "page": "Page 2 Title"})

    mock_loader_instance = MockWikipediaLoader.return_value
    mock_loader_instance.load.return_value = [mock_doc1, mock_doc2]

    query = "test query wikipedia"

    # Act
    result = wiki_search.run({"query": query})

    # Assert
    MockWikipediaLoader.assert_called_once_with(query=query, load_max_docs=2)
    mock_loader_instance.load.assert_called_once()

    assert "wiki_results" in result
    expected_output_doc1 = '<Document source="http://wiki.com/page1" page="Page 1 Title"/>\nContent of page 1\n</Document>'
    expected_output_doc2 = '<Document source="http://wiki.com/page2" page="Page 2 Title"/>\nContent of page 2\n</Document>'
    assert expected_output_doc1 in result["wiki_results"]
    assert expected_output_doc2 in result["wiki_results"]
    assert result["wiki_results"].count("---") == 1 # Separator between docs

@patch('the_bot.agents.tools.search.WikipediaLoader')
def test_wiki_search_no_results(MockWikipediaLoader):
    # Arrange
    mock_loader_instance = MockWikipediaLoader.return_value
    mock_loader_instance.load.return_value = []
    query = "query with no results"

    # Act
    result = wiki_search.run({"query": query})

    # Assert
    MockWikipediaLoader.assert_called_once_with(query=query, load_max_docs=2)
    assert result["wiki_results"] == ""

# --- Tavily Web Search Tests ---
@patch('the_bot.agents.tools.search.TavilySearchResults')
def test_web_search_success(MockTavilySearchResults):
    # Arrange
    mock_tavily_instance = MockTavilySearchResults.return_value
    mock_tavily_instance.invoke.return_value = [
        {"url": "http://example.com/result1", "title": "Result 1", "score": 0.9, "content": "Content for result 1"},
        {"url": "http://example.com/result2", "title": "Result 2", "score": 0.8, "content": "Content for result 2"},
    ]
    query = "test query tavily"

    # Act
    result = web_search.run({"query": query})

    # Assert
    MockTavilySearchResults.assert_called_once_with(max_results=3)
    mock_tavily_instance.invoke.assert_called_once_with({"query": query})

    assert "web_results" in result
    expected_output_doc1 = '<Document source="http://example.com/result1" title=Result 1 score="0.9"/>\nContent for result 1\n</Document>'
    expected_output_doc2 = '<Document source="http://example.com/result2" title=Result 2 score="0.8"/>\nContent for result 2\n</Document>'
    assert expected_output_doc1 in result["web_results"]
    assert expected_output_doc2 in result["web_results"]
    assert result["web_results"].count("---") == 1

@patch('the_bot.agents.tools.search.TavilySearchResults')
def test_web_search_no_results(MockTavilySearchResults):
    # Arrange
    mock_tavily_instance = MockTavilySearchResults.return_value
    mock_tavily_instance.invoke.return_value = []
    query = "tavily query no results"

    # Act
    result = web_search.run({"query": query})

    # Assert
    MockTavilySearchResults.assert_called_once_with(max_results=3)
    assert result["web_results"] == ""

# --- Arxiv Search Tests ---
@patch('the_bot.agents.tools.search.ArxivLoader')
def test_arxiv_search_success(MockArxivLoader):
    # Arrange
    mock_doc1 = Document(page_content="Long Arxiv content for paper 1...", metadata={"source": "http://arxiv.org/abs/1234.5678", "page": "1234.5678v1"})
    mock_doc2 = Document(page_content="Another Arxiv paper content...", metadata={"source": "http://arxiv.org/abs/9876.5432", "page": "9876.5432v2"})

    mock_loader_instance = MockArxivLoader.return_value
    mock_loader_instance.load.return_value = [mock_doc1, mock_doc2]

    query = "test query arxiv"

    # Act
    result = arxiv_search.run({"query": query})

    # Assert
    MockArxivLoader.assert_called_once_with(query=query, load_max_docs=3)
    mock_loader_instance.load.assert_called_once()

    assert "arxiv_results" in result
    # Content is truncated at 1000 chars for Arxiv, so we check for the beginning
    expected_output_doc1 = f'<Document source="http://arxiv.org/abs/1234.5678" page="1234.5678v1"/>\n{mock_doc1.page_content[:1000]}\n</Document>'
    expected_output_doc2 = f'<Document source="http://arxiv.org/abs/9876.5432" page="9876.5432v2"/>\n{mock_doc2.page_content[:1000]}\n</Document>'

    assert expected_output_doc1 in result["arxiv_results"]
    assert expected_output_doc2 in result["arxiv_results"]
    assert result["arxiv_results"].count("---") == 1

@patch('the_bot.agents.tools.search.ArxivLoader')
def test_arxiv_search_no_results(MockArxivLoader):
    # Arrange
    mock_loader_instance = MockArxivLoader.return_value
    mock_loader_instance.load.return_value = []
    query = "arxiv query no results"

    # Act
    result = arxiv_search.run({"query": query})

    # Assert
    MockArxivLoader.assert_called_once_with(query=query, load_max_docs=3)
    assert result["arxiv_results"] == ""
