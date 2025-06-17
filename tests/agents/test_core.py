import pytest
from unittest.mock import MagicMock, patch
import sys
import os
import tempfile
import shutil

# Assuming Agent class is in src.the_bot.agents.core
from src.the_bot.agents.core import Agent

# Minimal Agent initialization for testing _clean_answer and other isolated methods
# We'll mock dependencies heavily when testing __init__ or answer_question
@pytest.fixture
def basic_agent():
    # Mock dependencies for a basic agent instance not needing full LLM/tool setup
    with patch('src.the_bot.agents.core.Agent._init_logger', return_value=MagicMock()), \
         patch('src.the_bot.agents.core.Agent._load_default_tools', return_value=[]), \
         patch('src.the_bot.agents.core.Agent._discover_tools', return_value=[]), \
         patch('src.the_bot.agents.core.Agent._init_memory'), \
         patch('src.the_bot.agents.core.create_client'), \
         patch('src.the_bot.agents.core.SupabaseVectorStore'), \
         patch('src.the_bot.agents.core.HuggingFaceEmbeddings'), \
         patch('src.the_bot.agents.core.ChatGroq') as MockChatGroq, \
         patch('src.the_bot.agents.core.StateGraph') as MockStateGraph, \
         patch('builtins.open', new_callable=pytest.mock.mock_open, read_data="system prompt text"):

        # Mock the LLM instance directly to avoid API key errors during basic init
        mock_llm_instance = MockChatGroq.return_value
        mock_llm_instance.bind_tools.return_value = MagicMock() # Mock llm_with_tools

        # Mock StateGraph instance and its compile method
        mock_graph_instance = MockStateGraph.return_value
        mock_graph_instance.compile.return_value = MagicMock() # This is self.agent

        # Provide minimal valid args for __init__
        agent = Agent(
            model_type="groq", # Default, can be changed in specific tests
            api_key="dummy_key", # Needs a dummy key for ChatGroq
            supabase_url="http://dummy.supabase.co",
            supabase_service_key="dummy.supabase.key"
        )
        return agent

class TestAgentCleanAnswer:
    def test_clean_answer_string_no_change(self, basic_agent):
        assert basic_agent._clean_answer("Simple answer") == "Simple answer"

    def test_clean_answer_with_prefixes(self, basic_agent):
        assert basic_agent._clean_answer("The answer is Final Value") == "Final Value"
        assert basic_agent._clean_answer("Answer: 123") == "123"
        assert basic_agent._clean_answer("Final answer: Test") == "Test"
        assert basic_agent._clean_answer("The result is 42.5") == "42.5"
        assert basic_agent._clean_answer("To answer this question: Yes") == "Yes"
        assert basic_agent._clean_answer("Based on the information provided, No") == "No"
        assert basic_agent._clean_answer("According to the information: Maybe") == "Maybe"
        assert basic_agent._clean_answer("  Answer:   Indented   ") == "Indented"

    def test_clean_answer_with_quotes(self, basic_agent):
        assert basic_agent._clean_answer('"Quoted String"') == "Quoted String"
        assert basic_agent._clean_answer("'Single Quoted String'") == "Single Quoted String"
        assert basic_agent._clean_answer('" Answer: Quoted with prefix"') == "Quoted with prefix"
        assert basic_agent._clean_answer("The answer is 'Quoted after prefix'") == "Quoted after prefix"

    def test_clean_answer_float(self, basic_agent):
        assert basic_agent._clean_answer(123.45) == "123.45"
        assert basic_agent._clean_answer(123.0) == "123" # Integer value in float form
        assert basic_agent._clean_answer(1234.56) == "$1,234.56" # Currency like
        assert basic_agent._clean_answer(0.0) == "0"

    def test_clean_answer_int(self, basic_agent):
        assert basic_agent._clean_answer(42) == "42"
        assert basic_agent._clean_answer(0) == "0"
        assert basic_agent._clean_answer(-10) == "-10"

    def test_clean_answer_other_types(self, basic_agent):
        assert basic_agent._clean_answer(True) == "True"
        assert basic_agent._clean_answer(None) == "None" # Though None might not be a typical raw LLM answer
        assert basic_agent._clean_answer([1, 2]) == "[1, 2]"

    def test_clean_answer_empty_string(self, basic_agent):
        assert basic_agent._clean_answer("") == ""
        assert basic_agent._clean_answer("  ") == ""

    def test_clean_answer_prefix_only(self, basic_agent):
        assert basic_agent._clean_answer("Answer: ") == ""
        assert basic_agent._clean_answer("The answer is") == "The answer is" # No space after, not a full prefix match

# More tests for _discover_tools, __init__, and answer_question will be added subsequently.

# Create a temporary directory for mock tool modules
MOCK_TOOLS_DIR = "mock_tools_temp_dir"

# Mock tool function (simulating a @tool decorator by adding 'as_tool' attribute)
def mock_tool_function():
    pass
mock_tool_function.as_tool = True # This is how _discover_tools identifies tools

def another_mock_tool_function():
    pass
another_mock_tool_function.as_tool = True

def not_a_tool_function():
    pass

# Content for mock tool modules
MOCK_TOOL_MODULE_CONTENT = '''
from unittest.mock import MagicMock

# Simulating a @tool decorated function by adding 'as_tool' attribute
def actual_tool_in_module():
    pass
actual_tool_in_module.as_tool = True

def another_tool_in_module():
    pass
another_tool_in_module.as_tool = True

def not_a_tool_function_in_module():
    pass

# A class that might be present but shouldn't be picked up as a tool function
class SomeClass:
    pass
'''

MOCK_TOOL_MODULE_EMPTY_CONTENT = '''
# This module has no tools
def regular_function():
    pass
'''

@pytest.fixture
def tool_discovery_setup(basic_agent):
    # Create a temporary directory for mock modules
    temp_dir_path = os.path.join(os.getcwd(), MOCK_TOOLS_DIR)
    os.makedirs(temp_dir_path, exist_ok=True)

    # Create an __init__.py to make it a package
    with open(os.path.join(temp_dir_path, "__init__.py"), "w") as f:
        f.write("")

    # Create a mock tool module file
    with open(os.path.join(temp_dir_path, "sample_tool_module.py"), "w") as f:
        f.write(MOCK_TOOL_MODULE_CONTENT)

    # Create another mock tool module file (empty of tools)
    with open(os.path.join(temp_dir_path, "empty_tool_module.py"), "w") as f:
        f.write(MOCK_TOOL_MODULE_EMPTY_CONTENT)

    # Add this temp directory to sys.path so it can be imported
    sys.path.insert(0, os.getcwd())

    yield basic_agent # The agent instance for the test to use

    # Teardown: remove the temporary directory and from sys.path
    sys.path.pop(0)
    if os.path.exists(temp_dir_path): # Check if path exists before attempting to remove
        shutil.rmtree(temp_dir_path)


class TestAgentDiscoverTools:
    def test_discover_tools_single_module(self, tool_discovery_setup):
        agent = tool_discovery_setup
        # The module path should be relative to where python can import it from
        # e.g., MOCK_TOOLS_DIR.sample_tool_module
        discovered_tools = agent._discover_tools([f"{MOCK_TOOLS_DIR}.sample_tool_module"])

        assert len(discovered_tools) == 2
        tool_names = [tool.__name__ for tool in discovered_tools]
        assert "actual_tool_in_module" in tool_names
        assert "another_tool_in_module" in tool_names
        for tool in discovered_tools:
            assert hasattr(tool, "as_tool")

    def test_discover_tools_multiple_modules(self, tool_discovery_setup):
        agent = tool_discovery_setup
        # Create another module for this test case specifically
        with open(os.path.join(MOCK_TOOLS_DIR, "another_sample_module.py"), "w") as f:
            f.write("def specific_tool(): pass\\n") # Corrected newline character
            f.write("specific_tool.as_tool = True\\n") # Corrected newline character

        discovered_tools = agent._discover_tools([
            f"{MOCK_TOOLS_DIR}.sample_tool_module",
            f"{MOCK_TOOLS_DIR}.another_sample_module"
        ])

        assert len(discovered_tools) == 3 # 2 from sample_tool_module + 1 from another_sample_module
        tool_names = [tool.__name__ for tool in discovered_tools]
        assert "actual_tool_in_module" in tool_names
        assert "another_tool_in_module" in tool_names
        assert "specific_tool" in tool_names

    def test_discover_tools_module_with_no_tools(self, tool_discovery_setup):
        agent = tool_discovery_setup
        discovered_tools = agent._discover_tools([f"{MOCK_TOOLS_DIR}.empty_tool_module"])
        assert len(discovered_tools) == 0

    def test_discover_tools_mixed_modules(self, tool_discovery_setup):
        agent = tool_discovery_setup
        discovered_tools = agent._discover_tools([
            f"{MOCK_TOOLS_DIR}.sample_tool_module",
            f"{MOCK_TOOLS_DIR}.empty_tool_module"
        ])
        assert len(discovered_tools) == 2 # Only from sample_tool_module

    def test_discover_tools_non_existent_module(self, tool_discovery_setup):
        agent = tool_discovery_setup
        # Mock logger to check for import error logging
        agent.logger = MagicMock()
        discovered_tools = agent._discover_tools(["non_existent_module_xyz"])
        assert len(discovered_tools) == 0
        # Check that the logger was called with a message indicating failure
        agent.logger.debug.assert_any_call("Failed to import tools from non_existent_module_xyz: No module named 'non_existent_module_xyz'")

    def test_discover_tools_empty_list(self, tool_discovery_setup):
        agent = tool_discovery_setup
        discovered_tools = agent._discover_tools([])
        assert len(discovered_tools) == 0
