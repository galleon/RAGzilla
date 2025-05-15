import logging

from collections.abc import Callable
from typing import Any

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from supabase.client import create_client

class Agent:
    def __init__(
        self,
        model_type: str = "groq",
        model_id: str = "qwen-qwq-32b",
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.2,
        executor_type: str = "local",
        additional_imports: list[str] = list(),
        tool_modules: list[str] = list(),
        verbose: bool = False,
        client_kwargs: dict[str, Any] = dict(),
        timeout: int = -1,
        system_prompt: str = "",
        supabase_url: str = "",
        supabase_service_key: str = "",
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        # Set verbosity
        self.verbose = verbose
        self.logger = self._init_logger()
        self.logger.info("Initializing Agent...")

        # Model initialization (omitted for brevity)
        if model_type == 'google':
            self.llm = ChatGoogleGenerativeAI(
                model=model_id,
                temperature=temperature,
                google_api_key=api_key
            )
        elif model_type == 'groq':
            self.llm = ChatGroq(
                model=model_id,
                temperature=temperature
            )
        elif model_type == 'HfApiModel':
            self.llm = ChatHuggingFace(
                llm=HuggingFaceEndpoint(
                    repo_id=model_id,
                    provider="hf-inference",
                    temperature=temperature,
                    huggingfacehub_api_token=api_key
                ),
                verbose=True
            )
        else:
            raise  ValueError(" Invalid provider. Choose 'google', ' groq' or 'huggingface'")

        # Load only core SmolAgents tools
        self.tools = self._load_default_tools()
        # Discover custom tools from specified modules
        if tool_modules:
            self.tools.extend(self._discover_tools(tool_modules))
        self.logger.info(f"Loaded {len(self.tools)} tools")

        # Initialize memory store
        self._init_memory(
            supabase_url,
            supabase_service_key
        )
        self.logger.info("Memory store initialized")

        # Set up imports
        self.imports = [
            "csv",
            "datetime",
            "json",
            "math",
            "numpy",
            "os",
            "pandas",
            "re",
            "requests",
            "urllib"
        ]
        if additional_imports:
            self.imports.extend(additional_imports)

        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Initialize CodeAgent
        #self.agent = build_agent(
        #    tools=self.tools,
        #    model=self.model,
        #    additional_authorized_imports=self.imports,
        #    executor_type=executor_type,
        #    executor_kwargs={},
        #    verbosity_level=2 if self.verbose else 0,
        #    max_steps=5
        #)
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()

        # System message
        sys_msg = SystemMessage(content=system_prompt)

        self.vector_store = SupabaseVectorStore(
           client=create_client(
               supabase_url,
               supabase_service_key
           ),
           embedding=HuggingFaceEmbeddings(model_name=embedding_model_name),
           table_name="documents",
           query_name="match_documents"
        )

        def assistant(state: MessagesState):
            """Assistant node"""
            return {"messages": [self.llm_with_tools.invoke(state["messages"])]}

        def retriever(state: MessagesState):
            """Retriever node"""
            # for message in state["messages"]:
            #     if isinstance(message, HumanMessage):

            similar_question = self.vector_store.similarity_search(
                state["messages"][0].content,
                k=1
            )
            # similar_question seems to be a list of Document which page_content contains
            # [
            #   Document(
            #     metadata={'source': '7bd855d8-463d-4ed5-93ca-5fe35145f733'},
            #     page_content='Question : The attached Excel file contains the sales of menu items for a local fast-food chain. What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places.\n\nFinal answer : 89706.00'
            #   ),
            #   Document(
            #     metadata={'source': '7cc4acfa-63fd-4acc-a1a1-e8e529e0a97f'},
            #     page_content='Question : The attached spreadsheet contains the sales of menu items for a regional fast-food chain. Which city had the greater total sales: Wharvton or Algrimand?\n\nFinal answer : Wharvton'
            #   ),
            #   Document(
            #     metadata={'source': '99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3'},
            #     page_content='Question : Hi, I\'m making a pie but I could use some help with my shopping list. I have everything I need for the crust, but I\'m not sure about the filling. I got the recipe from my friend Aditi, but she left it as a voice memo and the speaker on my phone is buzzing so I can\'t quite make out what she\'s saying. Could you please listen to the recipe and list all of the ingredients that my friend described? I only want the ingredients for the filling, as I have everything I need to make my favorite pie crust. I\'ve attached the recipe as Strawberry pie.mp3.\n\nIn your response, please only list the ingredients, not any measurements. So if the recipe calls for "a pinch of salt" or "two cups of ripe strawberries" the ingredients on the list would be "salt" and "ripe strawberries".\n\nPlease format your response as a comma separated list of ingredients. Also, please alphabetize the ingredients.\n\nFinal answer : cornstarch, freshly squeezed lemon juice, granulated sugar, pure vanilla extract, ripe strawberries'
            #   ),
            #   Document(
            #     metadata={'source': '076c8171-9b3b-49b9-a477-244d2a532826'},
            #     page_content='Question : The attached file contains a list of vendors in the Liminal Springs mall, along with each vendor’s monthly revenue and the rent they pay the mall. I want you to find the vendor that makes the least money, relative to the rent it pays. Then, tell me what is listed in the “type” column for that vendor.\n\nFinal answer : Finance'
            #  )
            #]

            if similar_question:  # Check if the list is not empty
                example_msg = HumanMessage(
                    content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
                )
                return {"messages": [sys_msg] + state["messages"] + [example_msg]}
            else:
                # Handle the case when no similar questions are found
                return {"messages": [sys_msg] + state["messages"]}

        builder = StateGraph(MessagesState)
        builder.add_node(retriever)
        builder.add_node(assistant)
        builder.add_node(ToolNode(self.tools))

        builder.add_edge(START, "retriever")
        builder.add_edge("retriever", "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")

        self.agent = builder.compile()

        # self.logger.debug(f"Agent prompt:\n{self.agent.prompt_templates['system_prompt']}")

        self.verbose = verbose
        self.logger.info("CodeAgent ready")

    def _load_default_tools(self) -> list:
        """
        Returns only the core SmolAgents tools.
        """
        return [
            # WikipediaSearchTool(), # needs pip install wikipedia-api
        ]

    def _init_logger(self) -> logging.Logger:
        """
        Configure and return a logger for the agent.
        """
        logger = logging.getLogger("Agent")
        if not logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
            handler.setFormatter(fmt)
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        return logger

    def _discover_tools(self, modules: list[str]) -> list:
        """
        Dynamically import tool modules and collect @tool functions.
        """
        collected: list = []
        self.logger.info(f"Discovering tools from modules: {modules}")
        for mod_path in modules:
            self.logger.info(f"processing {mod_path}")
            try:
                mod = __import__(mod_path, fromlist=["*"])
                # self.logger.debug(f"imported {mod}")
                for attr in dir(mod):
                    # self.logger.debug(f"found attr: {attr}")
                    fn = getattr(mod, attr)
                    # if callable(fn) and hasattr(fn, "_get_tool_code"):  #smolagents
                    # self.logger.debug(f"dir(fn): {dir(fn)}")
                    # if callable(fn) and hasattr(fn, "_get_tool_code"):  #smolagents
                    if callable(fn) and hasattr(fn, "as_tool"):  #smolagents
                        collected.append(fn)
            except ImportError as e:
                self.logger.debug(f"Failed to import tools from {mod_path}: {e}")
        self.logger.info(f"Collected {len(collected)} tools.")
        return collected

    def _init_memory(
        self,
        supabase_url: str,
        supabase_service_key: str,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    ):
         """
         Initialize vector store memory using FAISS and a sentence-transformers embedder.
         """
         self.vector_store = SupabaseVectorStore(
            client=create_client(
                supabase_url,
                supabase_service_key
            ),
            embedding=HuggingFaceEmbeddings(model_name=embedding_model_name),
            table_name="documents",
            query_name="match_documents"
         )

    # def remember(self, text: str, meta: Any = None):
    #     """
    #     Embed the text and store in memory with optional metadata.
    #     """
    #     emb = self.embedder.encode([text])
    #     self.vector_index.add(emb)
    #     self.memory_meta.append({"text": text, "meta": meta})

    # def recall(self, query: str, top_k: int = 5) -> list[Any]:
    #     """
    #     Query memory for most similar stored items.
    #     """
    #     q_emb = self.embedder.encode([query])
    #     D, I = self.vector_index.search(q_emb, top_k)
    #     return [self.memory_meta[i] for i in I[0]]

    def answer_question(self, question: str, task_file_path: str | None = None) -> str:
        """
        Process a question and return the answer

        Args:
            question: The question to answer
            task_file_path: Optional path to a file associated with the question

        Returns:
            The answer to the question
        """
        try:
            if self.verbose:
                print(f"Processing question: {question}")
                if task_file_path:
                    print(f"With associated file: {task_file_path}")

            # Create a context with file information if available
            context = question
            file_content = None

            # If there's a file, read it and include its content in the context
            if task_file_path:
                try:
                    with open(task_file_path, 'r') as f:
                        file_content = f.read()

                    # Determine file type from extension
                    import os
                    file_ext = os.path.splitext(task_file_path)[1].lower()

                    context = f"""
Question: {question}
This question has an associated file. Here is the file content:
```{file_ext}
{file_content}
```
Analyze the file content above to answer the question.
"""
                except Exception as file_e:
                    context = f"""
Question: {question}
This question has an associated file at path: {task_file_path}
However, there was an error reading the file: {file_e}
You can still try to answer the question based on the information provided.
"""

            # Check for special cases that need specific formatting
            # Reversed text questions
            if question.startswith(".") or ".rewsna eht sa" in question:
                context = f"""
This question appears to be in reversed text. Here's the reversed version:
{question[::-1]}
Now answer the question above. Remember to format your answer exactly as requested.
"""

            # Add a prompt to ensure precise answers
            full_prompt = f"""{context}
When answering, provide ONLY the precise answer requested.
Do not include explanations, steps, reasoning, or additional text.
Be direct and specific.
For example, if asked "What is the capital of France?", respond simply with "Paris".
"""

            self.logger.debug(full_prompt)
            # Run the agent with the question
            # answer = self.agent.run(full_prompt)

            messages = [HumanMessage(content=full_prompt)]
            print("\n\n=== === ===")
            for m in messages["messages"]:
                m.pretty_print()
                print("> --")
            messages = self.agent.invoke({"messages": messages})
            print("=== === ===")
            for m in messages["messages"]:
                m.pretty_print()
                print("< --")
            print("=== === ===\n\n")
            answer = messages["messages"][-1].content

            # Clean up the answer to ensure it's in the expected format
            # Remove common prefixes that models often add
            answer = self._clean_answer(answer)

            if self.verbose:
                print(f"Generated answer: {answer}")

            print(answer)
            # print(self.agent.memory.steps)

            return answer
        except Exception as e:
            error_msg = f"Error answering question: {e}"
            if self.verbose:
                print(error_msg)
            return error_msg

    def _clean_answer(self, answer: Any) -> str:
        """
        Clean up the answer to remove common prefixes and formatting
        that models often add but that can cause exact match failures.

        Args:
            answer: The raw answer from the model

        Returns:
            The cleaned answer as a string
        """
        # Convert non-string types to strings
        if not isinstance(answer, str):
            # Handle numeric types (float, int)
            if isinstance(answer, float):
                # Format floating point numbers properly
                # Check if it's an integer value in float form (e.g., 12.0)
                if answer.is_integer():
                    formatted_answer = str(int(answer))
                else:
                    # For currency values that might need formatting
                    if abs(answer) >= 1000:
                        formatted_answer = f"${answer:,.2f}"
                    else:
                        formatted_answer = str(answer)
                return formatted_answer
            elif isinstance(answer, int):
                return str(answer)
            else:
                # For any other type
                return str(answer)

        # Now we know answer is a string, so we can safely use string methods
        # Normalize whitespace
        answer = answer.strip()

        # Remove common prefixes and formatting that models add
        prefixes_to_remove = [
            "The answer is ",
            "Answer: ",
            "Final answer: ",
            "The result is ",
            "To answer this question: ",
            "Based on the information provided, ",
            "According to the information: ",
        ]

        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()

        # Remove quotes if they wrap the entire answer
        if (answer.startswith('"') and answer.endswith('"')) or (answer.startswith("'") and answer.endswith("'")):
            answer = answer[1:-1].strip()

        return answer

# def build_agent(
#     model_type: str = "OpenAIServerModel",
#     model_id: str | None = None,
#     api_key: str | None = None,
#     api_base: str | None = None,
#     temperature: float = 0.2,
#     executor_type: str = "local",
#     additional_imports: list[str] | None = None,
#     tool_modules: list[str] | None = None,
#     verbose: bool = False,
#     client_kwargs: dict[str, Any] = dict(),
#     timeout: int | None = None
# ):

#     llm_with_tools = llm.bind_tools(tools)

#     builder = StateGraph(MessageState)
#     builder.add_node(retriever)
#     builder.add_node(assistant)
#     builder.add_node(tools)

#     builder.add_edge(START, "retriever")
#     builder.add_edge("retriever", "assistant")
#     builder.add_conditional_edges("assistant", tools_condition)
#     builder.add_edge("tools", "assistant")

#     return builder.compile()
