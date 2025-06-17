# The Bot

This project implements a Gradio-based web application that interacts with an AI agent to answer questions and evaluate its performance. The agent fetches tasks from a remote API, processes them, and submits the answers for scoring.

## Project Structure

```
.
├── .gitignore
├── README.md               # This file
├── pyproject.toml          # Project metadata and dependencies
└── src/
    └── the_bot/
        ├── __init__.py
        ├── agents/             # Contains the core agent logic and tools
        │   ├── __init__.py
        │   ├── core.py         # Main agent implementation
        │   ├── tools/          # Tools available to the agent
        │   └── utils.py        # Utility functions for agents
        ├── api/                # Code for interacting with external APIs
        │   ├── __init__.py
        │   └── client.py       # Client for the scoring API
        └── main.py             # Entry point for the Gradio application
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    # Install uv if you haven't already (see https://github.com/astral-sh/uv)
    # For example: curl -LsSf https://astral.sh/uv/install.sh | sh

    # Create a virtual environment and install dependencies
    uv venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    uv pip install -e .[dev]
    ```

3.  **Set up environment variables:**
    The application requires certain environment variables to be set. You can create a `.env` file in the project root and add your credentials there. The application will automatically load it if the `python-dotenv` package is installed.

    **Required:**
    *   At least one API key for the LLM provider. The application supports:
        *   `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN`: For Hugging Face models.
        *   `OPENAI_API_KEY`: For OpenAI models.
        *   `XAI_API_KEY`: For xAI models.
        *   `DASHSCOPE_API_KEY`: For DashScope models.
        *   `GEMINI_API_KEY`: For Google Gemini models.

    **Optional (defaults are provided or behavior changes):**
    *   `AGENT_MODEL_TYPE`: Specifies the model type. Defaults to `HfApiModel`. Other options include `groq`, `google`.
    *   `AGENT_MODEL_ID`: The specific model ID to use (e.g., `gpt-4o`, `meta-llama/Llama-3.3-70B-Instruct`). The default depends on the `AGENT_MODEL_TYPE`.
    *   `AGENT_TEMPERATURE`: Sets the creativity of the model. Defaults to `0.2`.
    *   `AGENT_VERBOSE`: Set to `true` for detailed logging from the agent. Defaults to `false`.
    *   `AGENT_API_BASE`: For OpenAI-compatible APIs, sets a custom base URL.
    *   `DASHSCOPE_API_BASE`: Custom base URL for DashScope. Defaults to `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`.
    *   `XAI_API_BASE`: Custom base URL for xAI. Defaults to `https://api.x.ai/v1`.
    *   `SUPABASE_URL`: URL for Supabase integration (optional).
    *   `SUPABASE_SERVICE_KEY`: Service key for Supabase (optional).
    *   `SYSTEM_PROMPT`: A custom system prompt for the agent.
    *   `SPACE_ID`: If deploying to Hugging Face Spaces, this is your Space ID.
    *   `SPACE_HOST`: If deploying to Hugging Face Spaces, this is your Space host.

    The `src/the_bot/main.py` script will print the status of these environment variables (whether they are set or not, without revealing the values) when it starts.

## Running the Application

Once the setup is complete, you can run the Gradio application:

```bash
python src/the_bot/main.py
```

This will start a local web server, and you can access the application by navigating to the URL displayed in your terminal (usually `http://127.0.0.1:7860` or `http://0.0.0.0:7860`).

## Using the Application

The Gradio interface will provide the following:

1.  **Login Button:** You'll need to log in with your Hugging Face account to submit your agent's answers for evaluation.
2.  **Run Evaluation & Submit All Answers Button:** Clicking this button will:
    *   Fetch all tasks (questions and optional associated files) from the scoring API.
    *   Initialize the AI agent based on your environment variable configuration.
    *   Run the agent on each task to generate answers.
    *   Submit all answers to the scoring API.
3.  **Status:** Displays the status of the submission and the overall score.
4.  **Results:** A table showing each task ID, the question, and the agent's generated answer.
5.  **Local Evaluation:** Shows the evaluation score based on local checking if available (the primary score comes from the server after submission).

The application also creates a `main.log` file with detailed logs.
