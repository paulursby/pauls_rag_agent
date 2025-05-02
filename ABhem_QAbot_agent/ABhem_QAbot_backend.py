# Play around with Agents using LangGraph and TavilySearch
# This is backend of the AB Hem Q&A bot/chat agent

"""Required packages
pip install langgraph langchain-core langchain-openai langchain-community tavily-python langgraph-checkpoint-sqlite
"""

import logging
import operator
from typing import Annotated, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

# Setting default logging level for modules not having a specified log level defined
logging.basicConfig(level=logging.INFO)
# Configure logging for this module
logger = logging.getLogger("paul_recept_QAbot_backend_logger")
logger.setLevel(logging.DEBUG)

# Content in ~/.bashrc to activate Langsmith tracing
# export LANGSMITH_TRACING=true
# export LANGSMITH_ENDPOINT="https://eu.api.smith.langchain.com"
# export LANGSMITH_API_KEY="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"
# export LANGSMITH_PROJECT="default-pauls-rags"

# A context manager for tracing a specific block of code.
# with tracing_v2_enabled():

# Define the search tool
tool = TavilySearchResults(
    max_results=10,
    include_domains=["ab-hem.se"],
    search_depth="advanced",
    # include_raw_content=True,
)


# Create Agent State functionality
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# Define the Agent Graph
class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t["name"]].invoke(t["args"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )
        print("Back to the model!")
        return {"messages": results}


# Function to run the agent and return the answer
def run_agent(system_prompt, user_query):
    """
    Run the agent with the given system prompt and user query

    Args:
        system_prompt (str): The system prompt for the assistant
        user_query (str): The user's question

    Returns:
        str: The answer to the user's query
    """
    # Define model
    model = ChatOpenAI(model="gpt-4o")

    # Create messages
    messages = [HumanMessage(content=user_query)]

    # Final answer placeholder
    user_answer = ""

    with SqliteSaver.from_conn_string(":memory:") as memory:
        # Creating the Agent
        abot = Agent(model, [tool], system=system_prompt, checkpointer=memory)

        # Create the Graph stream
        thread = {"configurable": {"thread_id": "1"}}

        # Track the final non-tool message
        final_message = None

        for event in abot.graph.stream({"messages": messages}, thread):
            for v in event.values():
                for message in v["messages"]:
                    # Print message info for debugging
                    print(f"\nMessage type: {type(message).__name__}")
                    print(f"Content: {message.content}")

                    # If there are tool calls, print those separately
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        print(f"Tool calls: {message.tool_calls}")
                    else:
                        # Save the message if it's not a tool call message
                        final_message = message

    # Get the final answer
    if final_message:
        user_answer = final_message.content
        print(f"\nAnswer to user query:\n{user_answer}")

    return user_answer


# This section will only run if the script is executed directly
if __name__ == "__main__":
    # Default system prompt
    DEFAULT_PROMPT = """You are a smart research assistant. Use the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """

    # Example usage
    query = "Vad är telefonnumret för felanmälan till AB-hem?"
    result = run_agent(DEFAULT_PROMPT, query)
    print("\nFinal answer:", result)
