import json
import logging
import os

import gradio as gr
import requests
import pandas as pd

from the_bot.agents.core import Agent
from the_bot.agents.utils import get_score

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

logger = logging.getLogger(__file__)
fmt = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler()
sh.setFormatter(fmt)
logger.addHandler(sh)

fh = logging.FileHandler("main.log")
fh.setFormatter(fmt)
logger.addHandler(fh)

def debug_environment():
    """Print which API vars are set (values redacted)."""
    for var in [
        "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
        "OPENAI_API_KEY", "XAI_API_KEY",
        "AGENT_MODEL_TYPE", "AGENT_MODEL_ID",
        "AGENT_TEMPERATURE", "AGENT_VERBOSE",
        "DASHSCOPE_API_KEY", "GEMINI_API_KEY",
        "SUPABASE_URL", "SUPABASE_SERVICE_KEY"
    ]:
        status = "[SET]" if os.getenv(var) else "[NOT SET]"
        print(f"{var}: {status}")

class AgentWrapper:
    def __init__(self):
        # Load .env if available
        try:
            import dotenv
            dotenv.load_dotenv()
            print("Loaded .env")
        except ImportError:
            pass

        debug_environment()

        # Gather config from env
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        openai_key = os.getenv("OPENAI_API_KEY")
        xai_key = os.getenv("XAI_API_KEY")
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        model_type = os.getenv("AGENT_MODEL_TYPE", "HfApiModel")
        # model_id = os.getenv("AGENT_MODEL_ID", "gpt-4o")
        temperature = float(os.getenv("AGENT_TEMPERATURE", "0.2"))
        verbose = os.getenv("AGENT_VERBOSE", "false").lower() == "true"
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
        system_prompt=os.getenv("SYSTEM_PROMPT")

        # Decide which credentials to use
        agent_kwargs = {
            "model_type": model_type,
            "temperature": temperature,
            "executor_type": "local",
            "verbose": verbose,
            "tool_modules": ["the_bot.agents.tools"],
            "system_prompt": system_prompt,
            "supabase_url": supabase_url,
            "supabase_service_key": supabase_service_key
        }
        if model_type == "groq":
            # OpenAI | xai | dashscope
            if dashscope_key:
                agent_kwargs["api_key"] = dashscope_key
                agent_kwargs["api_base"] = os.getenv("DASHSCOPE_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
                agent_kwargs["model_id"] = os.getenv("AGENT_MODEL_ID")
            elif xai_key:
                agent_kwargs["api_key"] = xai_key
                agent_kwargs["api_base"] = os.getenv("XAI_API_BASE", "https://api.x.ai/v1")
            elif openai_key:
                agent_kwargs["api_key"] = openai_key
                agent_kwargs["api_base"] = os.getenv("AGENT_API_BASE", None)
            else:
                raise RuntimeError("No API credentials found for OpenAI")
        elif gemini_key and model_type == "google":
            agent_kwargs["api_key"] = gemini_key
            agent_kwargs["model_id"] = os.getenv("AGENT_MODEL_ID", "gemini-2.5-flash-preview-04-17")
        elif hf_token and model_type == "HfApiModel":
            agent_kwargs["api_key"] = hf_token
            agent_kwargs["model_id"] = os.getenv("AGENT_MODEL_ID", "meta-llama/Llama-3.3-70B-Instruct")
        else:
            raise RuntimeError("No API credentials found for Hugging Face or OpenAI compatible services")

        # Instantiate
        logger.info(f"Initializing Agent with: {agent_kwargs}")
        self.agent = Agent(**agent_kwargs)
        logger.info(f"Initialized Agent: {agent_kwargs}")
        print(f"Initialized Agent: {agent_kwargs}")

    def __call__(self, question: str, file_name: str) -> str:
        if not question.strip():
            return "Please provide a question."
        try:
            return self.agent.answer_question(question, file_name)
        except Exception as e:
            logger.debug("Error in agent:", e)
            return "Agent error—see logs."

def run_and_submit_all(profile: gr.OAuthProfile | None):
    # Authentication check
    username = None
    if profile:
        username = profile.username

    # fetch tasks
    questions_url = f"{DEFAULT_API_URL}/questions"
    try:
        resp = requests.get(questions_url, timeout=15)
        resp.raise_for_status()
        tasks = resp.json()
    except Exception as e:
        return f"Failed to fetch tasks: {e}", None

    # fetch files attached to tasks
    for task in tasks:
        if task["file_name"]:
            resp = requests.get(f"{DEFAULT_API_URL}/files/{task['task_id']}")
            if resp.status_code == 200:
                with open(f"{task['file_name']}", "wb") as f:
                    f.write(resp.content)
            else:
                logger.info(f"Failed to retrieve the file. Status code: {resp.status_code}")

    # Instantiate agent once
    try:
        agent = AgentWrapper()
    except Exception as e:
        return f"Agent initialization failed: {e}", None

    logger.debug("Agent initialized.")

    results = []
    payload = []
    for task in tasks:
        tid = task.get("task_id")
        logger.debug(f"Task ID: {tid}")
        q = task.get("question", "")
        logger.debug(f"Question: {q}")
        file_name = task.get("file_name", "")
        if not tid or not q:
            continue
        ans = agent(q, file_name)
        logger.debug(f"Answer: {ans}")
        payload.append({"task_id": tid, "submitted_answer": ans})
        results.append({"task_id": tid, "question": q, "answer": ans})

    if not payload:
        return "No answers generated.", pd.DataFrame(results)

    # Submit
    submit_url = f"{DEFAULT_API_URL}/submit"
    data = {"username": username, "agent_code": f"https://huggingface.co/spaces/{os.getenv('SPACE_ID')}/tree/main", "answers": payload}

    res = get_score(data)
    status_txt = (
        f"User: {res.get('username')}  "
        f"Score: {res.get('score')}%  "
        f"Correct: {res.get('correct_count')}/{res.get('total_attempted')}"
    )

    if username:
        try:
            resp = requests.post(submit_url, json=data, timeout=60)
            resp.raise_for_status()
            res = resp.json()
            status = (
                f"User: {res.get('username')}  "
                f"Score: {res.get('score')}%  "
                f"Correct: {res.get('correct_count')}/{res.get('total_attempted')}"
            )
        except Exception as e:
            status = f"Submission failed: {e}"

        return status, pd.DataFrame(results), status_txt
    else:
        return "Please log in to submit.", pd.DataFrame(results), status_txt

# --- Gradio UI ---

with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.Markdown("Log in and click below to run & submit all tasks.")

    gr.LoginButton()
    run_btn = gr.Button("Run Evaluation & Submit All Answers")
    status_out = gr.Textbox(label="Status", interactive=False)
    results_tbl = gr.DataFrame(label="Results")
    status_txt = gr.Textbox(label="Local Evaluation", interactive=False)

    run_btn.click(fn=run_and_submit_all, outputs=[status_out, results_tbl, status_txt])

if __name__ == "__main__":
    logger.info("Launching Agent Gradio app…")

    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")
    demo.launch(server_name="0.0.0.0", debug=True, share=False)
