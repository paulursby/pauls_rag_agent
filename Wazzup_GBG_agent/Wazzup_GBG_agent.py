# Play around with Agents using LangGraph and TavilySearch
# This is backend of the Agent

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
tool = TavilySearchResults(max_results=2, include_domains=["ab-hem.se"])


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


# Define System prompt
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

prompt2 = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
Look up information only using.
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

# Define model
model = ChatOpenAI(model="gpt-4o")

with SqliteSaver.from_conn_string(":memory:") as memory:
    # Creating the Agent
    abot = Agent(model, [tool], system=prompt, checkpointer=memory)

    # Create the user query
    messages = [
        HumanMessage(content="Vad är telefonnumret för felanmälan till AB-hem?")
    ]

    # Create the Graph stream
    thread = {"configurable": {"thread_id": "1"}}

    """
    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(v["messages"])
    """

    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            # Print only the message content
            for message in v["messages"]:
                print(f"\nMessage type: {type(message).__name__}")
                print(f"Content: {message.content}")

                # If there are tool calls, print those separately
                if hasattr(message, "tool_calls") and message.tool_calls:
                    print(f"Tool calls: {message.tool_calls}")

user_answer = message.content
print(f"\nAnswer to user query:\n{user_answer}")
