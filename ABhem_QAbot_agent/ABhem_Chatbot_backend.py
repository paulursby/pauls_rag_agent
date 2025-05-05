# Play around with Agents using LangGraph and TavilySearch
# This is backend of the AB Hem Q&A bot/chat agent

"""
Required packages to install:
pip install langgraph langchain-core langchain-openai langchain-community tavily-python langgraph-checkpoint-sqlite
"""

import logging
import operator
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from typing import Annotated, Optional, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (
    AnyMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
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
    user_email_address: Optional[str]
    email_sent: Optional[bool]


# Define the Agent Graph
class Agent:
    def __init__(self, model, tools, checkpointer, system_prompt=""):
        self.system_prompt = system_prompt
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_node("collect_email_address", self.collect_email_address)
        graph.add_node("send_email", self.send_email)

        # Add edges
        graph.add_conditional_edges(
            "llm",
            self.route_next_step,
            {"action": "action", "email": "collect_email_address", "end": END},
        )
        graph.add_edge("action", "llm")
        graph.add_edge("collect_email_address", "send_email")
        graph.add_edge("send_email", END)

        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def route_next_step(self, state: AgentState) -> str:
        """
        Check the latest message to determine if tool is required, found an answer,
        or need email collection
        """

        latest_message = state["messages"][-1]

        # Parse the content for indicators
        content = (
            latest_message.content.lower() if hasattr(latest_message, "content") else ""
        )

        # If tool calls, next step is action
        if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
            return "action"

        # If no answer is found, next step is email collection
        if "vi kan tyvärr inte svara på din fråga nu" in content:
            return "email"

        # Otherwise an answer is found, next step is the END
        return "end"

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

    def collect_email_address(self, state: AgentState):
        """
        Handles email collection process
        """
        # TODO: Why is this required? Can it be skipped?

        user_email_adress = input(
            "Vänligen ange din e-postadress så återkommer vi med svar så snart som möjligt: "
        )
        print(f"Din e-postadress: {user_email_adress}")
        email_request = ChatMessage(
            role="system_event",
            content="User has provided an email address with valid format",
        )

        # Add this message to conversation
        return {"messages": [email_request], "user_email_address": user_email_adress}

    def send_email(self, state: AgentState):
        """
        Send email to backoffice to manually answer on user question
        """
        back_office_email_address = "paulursby@hotmail.com"
        # back_office_email_address = "ulrik@baard.se"

        # Get user query and email from state
        user_email_address = state.get("user_email_address", "No email provided")

        # TODO: Remove
        print("\nAll messages in state are printed:")
        for message in state["messages"]:
            print(f"Message type: {message.type}\nMessage: {message}\n")

        # Extract the original user query, which is the first HumanMessage in the
        # messages list
        user_query = "No query found"
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                user_query = message.content
                break

        # Create email
        email = EmailMessage()
        email["From"] = "paulursby@gmail.com"  # Replace with your sending email
        email["To"] = back_office_email_address
        email["Subject"] = (
            f"AB Hem Q&A Request - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        body = f"""
        A user has submitted a question that could not be answered automatically.
        
        User Query: {user_query}
        
        User Email: {user_email_address}
        
        Please respond directly to the user at their email address with an answer to their query.
        
        Thank you,
        AB Hem Q&A Bot
        """
        email.set_content(body)

        # Send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            try:
                server.login("paulursby@gmail.com", "enrn donl mbje rswg")
                server.send_message(email)
                print("Email sent successfully")
                # Add a message that email is succesfully sent
                confirmation = ChatMessage(
                    role="system_event",
                    content="Email message is succesfully sent to back-office",
                )

                return {"messages": [confirmation], "email_sent": True}
            except Exception as e:
                print(f"Failed to send email: {e}")

                # Add a message that email sent failed
                confirmation = ChatMessage(
                    role="system_event",
                    content="Email message sent to back-office failed",
                )

                return {"messages": [confirmation], "email_sent": False}


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

    # Initial state
    initial_state = {
        "messages": messages,
        "user_email_address": None,
        "email_sent": False,
    }

    # Final answer placeholder
    user_answer = ""

    with SqliteSaver.from_conn_string(":memory:") as memory:
        # Creating the Agent
        abot = Agent(model, [tool], system_prompt=system_prompt, checkpointer=memory)

        # Create the Graph stream
        thread = {"configurable": {"thread_id": "1"}}

        # Track the final non-tool message
        final_message = None

        for event in abot.graph.stream(initial_state, thread):
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
    DEFAULT_PROMPT = """You are a smart research assistant. Use the search engine to look up information. 
    You are allowed to make multiple calls (either together or in sequence). 
    Only look up information when you are sure of what you want. 
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    All questions are about the company AB-hem. 
    Search for an answer on AB Hems homepage only, which is https://ab-hem.se/. 
    If no answer is found on AB Hems homepage, then answer with this message: 
    "Vi kan tyvärr inte svara på din fråga nu."
    """

    # Example usage
    # query = "Vad är telefonnumret för felanmälan till AB-hem?"
    query = "Vem äger AB Hem?"
    result = run_agent(DEFAULT_PROMPT, query)
    print("\nFinal answer:", result)
