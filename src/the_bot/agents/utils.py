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



def get_score(data:dict) -> dict:
    ref = {
        "8e867cd7-cff9-4e6c-867a-ff5ddc2550be": "3",
        "a1e91b78-d3d8-4675-bb8d-62741b4b68a6": "3",
        "2d83110e-a098-4ebb-9987-066c06fa42d0": "Right",
        "cca530fc-4052-43b2-b130-b30968d8aa44": "Rd5",
        "4fc2f1ae-8625-45b5-ab34-ad4433bc21f8": "FunkMonk",
        "6f37996b-2ac7-44b0-8e68-6d28256631b4": "b, e",
        "9d191bce-651d-4746-be2d-7ef8ecadb9c2": "Extremely",
        "cabe07ed-9eca-40ea-8ead-410ef5e83f91": "Louvrier",
        "3cef3a44-215e-4aed-8e3b-b1e3f08063b7": "broccoli, celery, fresh basil, lettuce, sweet potatoes",
        "99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3": "cornstarch, freshly squeezed lemon juice, granulated sugar, pure vanilla extract, ripe strawberries",
        "305ac316-eef6-4446-960a-92d80d542f82": "Wojciech",
        "f918266a-b3e0-4914-865d-4faa564f1aef": "0",
        "3f57289b-8c60-48be-bd80-01f8099ca449": "519",
        "1f975693-876d-457b-a649-393859e79bf3": "132,133,134,197,245",
        "840bfca7-4f7b-481a-8794-c560c340185d": "80GSFC21M002",
        "bda648d7-d618-4883-88f4-3466eabd860e": "Saint Petersburg",
        "cf106601-ab4f-4af9-b045-5295fe67b37d": "CUB",
        "a0c07678-e491-4bbc-8f0b-07405144218f": "Yoshida, Uehara",
        "7bd855d8-463d-4ed5-93ca-5fe35145f733": "89706.00",
        "5a0c1adf-205e-4841-a666-7c3ef95def9d": "Claus",
    }

    correct_count = 0
    total_attempted = 0

    answers:dict = data.get("answers", {})
    print(answers, ref.keys(), ref.values())

    for item in answers:
        tid, answer = item["task_id"], item["submitted_answer"]
        total_attempted += 1
        if tid in ref:
            print(f"{answer} === {ref.get(tid, '')}")
            if answer == ref.get(tid, ""):
                correct_count += 1
                print("OK")

    score = 100 * correct_count / len(ref.keys())

    return {
        "username": data["username"],
        "correct_count": correct_count,
        "total_attempted": total_attempted,
        "score": score
    }
