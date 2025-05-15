#
# Wrapping a Smolagent tool for langgraph
#
# class SmolagentToolWrapper(BaseTool):
#     """Wrapper for smolagents tools to make them compatible with LangChain."""

#     wrapped_tool: object = Field(description="The wrapped smolagents tool")

#     def __init__(self, tool):
#         """Initialize the wrapper with a smolagents tool."""
#         super().__init__(
#             name=tool.name,
#             description=tool.description,
#             return_direct=False,
#             wrapped_tool=tool
#         )

#     def _run(self, query: str) -> str:
#         """Use the wrapped tool to execute the query."""
#         try:
#             # For WikipediaSearchTool
#             if hasattr(self.wrapped_tool, 'search'):
#                 return self.wrapped_tool.search(query)
#             # For DuckDuckGoSearchTool and others
#             return self.wrapped_tool(query)
#         except Exception as e:
#             return f"Error using tool: {str(e)}"

#     def _arun(self, query: str) -> str:
#         """Async version - just calls sync version since smolagents tools don't support async."""
#         return self._run(query)
