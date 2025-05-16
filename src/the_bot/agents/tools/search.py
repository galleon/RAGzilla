from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders  import WikipediaLoader
from langchain_community.document_loaders  import ArxivLoader
from langchain_core.tools import tool

@tool
def wiki_search(query: str) -> dict:
    """
    Search Wikipedia for a query and return maximum 2 results.

    Args:
        query: The search query.
    """
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"wiki_results": formatted_search_docs}


@tool
def web_search(query: str) -> dict:
    """
    Search Tavily for a query and return maximum 3 results.

    Args:
        query: The search query.
    """
    search_docs = TavilySearchResults(max_results=3).invoke({"query": query})

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.get("url", "")}" title={doc.get(" title", "")} score="{doc.get("score", 0.0)}"/>\n{doc.get("content", "")}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"web_results": formatted_search_docs}


@tool
def arxiv_search(query: str) -> dict:
    """
    Search Arxiv for a query and return maximum 3 result.

    Args:
        query: The search query.
    """
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"arxiv_results": formatted_search_docs}


# from typing import Any, Optional
# from smolagents.tools import Tool
# import duckduckgo_search

# class DuckDuckGoSearchTool(Tool):
#     name = "web_search"
#     description = "Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results."
#     inputs = {'query': {'type': 'string', 'description': 'The search query to perform.'}}
#     output_type = "string"

#     def __init__(self, max_results=10, **kwargs):
#         super().__init__()
#         self.max_results = max_results
#         try:
#             from duckduckgo_search import DDGS
#         except ImportError as e:
#             raise ImportError(
#                 "You must install package `duckduckgo_search` to run this tool: for instance run `pip install duckduckgo-search`."
#             ) from e
#         self.ddgs = DDGS(**kwargs)

#     def forward(self, query: str) -> str:
#         results = self.ddgs.text(query, max_results=self.max_results)
#         if len(results) == 0:
#             raise Exception("No results found! Try a less restrictive/shorter query.")
#         postprocessed_results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
#         return "## Search Results\n\n" + "\n\n".join(postprocessed_results)
