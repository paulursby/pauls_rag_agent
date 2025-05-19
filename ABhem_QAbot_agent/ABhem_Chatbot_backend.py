"""
This is the backend module of the AB Hem Chat bot agent, using e.g ChatOpenAI,
LangChain, LangGraph and TavilySearch capabilities.

Precondition is that these packages are pip installed:
pip install langgraph langchain-core langchain-openai langchain-community
tavily-python langgraph-checkpoint-sqlite
"""

import logging
import operator
import os
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
from lib.helper_functions import is_valid_email

# Define the directory where you want to store the log file
log_directory = "/home/paulur/python-projects/pauls_rag_agent/Log_files"

# Create the directory if it doesn't exist
os.makedirs(log_directory, exist_ok=True)

# Set up the full path to the log file
log_filepath = os.path.join(log_directory, "ABhem_Chatbot_backend.log")

# Configure logging with the full filepath
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # filename=log_filepath,
)

# Configure logging for this module
logger = logging.getLogger("ABhem_Chatbot_backend_logger")
logger.setLevel(logging.DEBUG)

"""
Content in ~/.bashrc to activate Langsmith tracing
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT="https://eu.api.smith.langchain.com"
export LANGSMITH_API_KEY="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"
export LANGSMITH_PROJECT="default-pauls-rags"

A context manager for tracing a specific block of code.
with tracing_v2_enabled():
"""

# Define the search tool
tool = TavilySearchResults(
    max_results=10,
    include_domains=["ab-hem.se"],
    search_depth="advanced",
    # TODO: Shall this be tried out?
    # include_raw_content=True,
)


# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_email_address: Optional[str]
    email_sent: Optional[bool]


# Define the Agent Graph
class Agent:
    """
    A conversational agent that processes user queries using a state graph workflow.

    The Agent uses a state graph to manage conversation flow, leveraging an LLM model
    for generating responses and tools for performing actions. It can route conversations
    to different paths based on the content of messages, collect email addresses, and
    forward unresolved queries to backend support staff.

    Attributes:
        system_prompt (str): Initial prompt to provide context to the LLM.
        graph: The compiled state graph that defines the agent's workflow.
        tools (dict): Dictionary mapping tool names to tool objects.
        model: The language model used for generating responses.
    """

    def __init__(self, model, tools, checkpointer, system_prompt=""):
        """
        Initialize the Agent with a model, tools, and optional system prompt.

        Args:
            model: The language model to use for responses.
            tools: List of tools the agent can use to perform actions.
            checkpointer: Object for saving and loading the state of the agent.
            system_prompt (str, optional): Initial prompt to provide context to the LLM.
                Defaults to empty string.
        """
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
        """
        Call the OpenAI language model with the current state's messages.

        Prepends the system prompt to the messages if it exists, then invokes
        the language model to generate a response.

        Args:
            state (AgentState): The current state containing messages.

        Returns:
            dict: Dictionary with updated messages including the model's response.
        """
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages

        message = self.model.invoke(messages)
        logger.info("LLM is called and a response is received.")

        return {"messages": [message]}

    def route_next_step(self, state: AgentState) -> str:
        """
        Determine the next step in the agent workflow based on the latest message.

        Analyzes the content of the latest message to decide whether to:
        - Execute a tool call
        - Collect user email for unresolved queries
        - End the conversation if an answer is provided

        Args:
            state (AgentState): The current state containing messages.

        Returns:
            str: The name of the next node in the workflow graph to execute.
        """
        logger.info("Next step in the Agent flow will be decided.")
        # Fetch latest message received in the Agent
        latest_message = state["messages"][-1]

        # Parse the content for indicators
        content = (
            latest_message.content.lower() if hasattr(latest_message, "content") else ""
        )

        # If tool calls, next step is action
        if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
            logger.info("Next step is a tool call.")
            return "action"

        # If no answer is found, next step is email collection
        if "vi kan tyvärr inte svara på din fråga nu" in content:
            logger.info(
                "Next step is to collect users email address, since no answer is found."
            )
            return "email"

        # Otherwise an answer is found, next step is the END
        logger.info("Next step is to exit the Agent flow, since an answer is found.")
        return "end"

    def take_action(self, state: AgentState):
        """
        Execute tool calls based on the latest message in the state.

        Processes all tool calls from the latest message, invoking the appropriate
        tools and collecting their responses.

        Args:
            state (AgentState): The current state containing messages with tool calls.

        Returns:
            dict: Dictionary with updated messages including tool responses.
        """
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            logger.info(f"Calling tool {t}.")
            # Call the tavily_search tool and store the response
            result = self.tools[t["name"]].invoke(t["args"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )
        logger.info("Tool has given a response and next step is calling the LLM.")

        return {"messages": results}

    def collect_email_address(self, state: AgentState):
        """
        Collect and validate the user's email address for follow-up on unresolved queries.

        Prompts the user to input their email address with validation, allowing multiple
        attempts. The function supports internationalized feedback in Swedish and
        provides detailed validation error messages to the user.

        Args:
            state (AgentState): The current state containing messages.

        Returns:
            dict: Dictionary containing:
                - messages (list): List with a single ChatMessage indicating the outcome
                  (either successful validation or too many failed attempts)
                - user_email_address (str): The collected email address if valid;
                  otherwise, the last attempt (even if invalid)
        """
        # TODO: max_attempt shall be config parameter
        max_attempts = 3
        for attempt in range(max_attempts):
            user_email_adress = input(
                "Vänligen ange din e-postadress så återkommer vi med svar så snart som möjligt: "
            )
            is_valid, message = is_valid_email(user_email_adress)

            if is_valid:
                email_request = ChatMessage(
                    role="system_event",
                    content="User has provided a valid email address.",
                )
                logger.info("User has provided a valid email address.")
                break
            else:
                attempts_left = max_attempts - attempt - 1
                print(f"Ogiltig e-postadress. {message}")
                if attempts_left > 0:
                    print(f"Du har {attempts_left} försök kvar.")
                logger.debug(
                    f"Entered email address: {user_email_adress} is invalid. {message}"
                )

            if attempt == (max_attempts - 1):
                user_email_adress = None
                email_request = ChatMessage(
                    role="system_event",
                    content="User has provided invalid email addresses too many times.",
                )
                logger.info("User has provided invalid email addresses too many times.")

        return {"messages": [email_request], "user_email_address": user_email_adress}

    def send_email(self, state: AgentState):
        """
        Send an email to the back office with the user's query for manual handling.

        Creates and sends an email containing the user's original query and their
        email address for follow-up. Handles success and failure scenarios.

        Args:
            state (AgentState): The current state containing messages and user email.
                Expected to contain:
                - 'user_email_address': String with user's email (from
                collect_email_address)
                - 'messages': List of message objects including at least one
                HumanMessage

        Returns:
            dict: Dictionary containing:
                - messages (list): List with a single ChatMessage indicating the outcome
                - email_sent (bool): True if email was sent successfully, False
                otherwise

        Error Handling:
            - Missing/invalid email address: Returns with email_sent=False and
            appropriate message
            - SMTP/connection errors: Catches exceptions, logs errors, returns with
            email_sent=False
            - Missing user query: Uses "No query found" as fallback if no HumanMessage
            is found
        """
        # TODO: Shall be config parameter
        back_office_email_address = "paulursby@hotmail.com"
        # back_office_email_address = "ulrik@baard.se"

        # Get user query and email from state
        user_email_address = state.get("user_email_address", "No user email provided")

        # Extract the original user query, which is the first HumanMessage in the
        # messages list
        user_query = "No query found"
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                user_query = message.content
                break

        # No user email can be sent due to no valid user email address provided
        if user_email_address == "No user email provided" or not user_email_address:
            # Add a state message that no user email address is provided
            confirmation = ChatMessage(
                role="system_event",
                content="Email message with user query can not be sent due to no valid "
                "user email address provided",
            )
            logger.error(
                "Email message with user query can not be sent due to no valid user "
                "email address provided."
            )

            return {"messages": [confirmation], "email_sent": False}

        # Create email if valid email address is entered by user
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

        # Send email and handle failure scenario
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            try:
                server.login("paulursby@gmail.com", "enrn donl mbje rswg")
                server.send_message(email)

                # Add a state message that email is succesfully sent
                confirmation = ChatMessage(
                    role="system_event",
                    content="Email message with user query is succesfully sent to "
                    "back-office",
                )
                logger.info(
                    "Email message with user query is succesfully sent to back-office."
                )

                return {"messages": [confirmation], "email_sent": True}
            except Exception as e:
                # Add a state message that sent email failed
                confirmation = ChatMessage(
                    role="system_event",
                    content="Email message with user query sent to back-office failed",
                )
                logger.error(
                    "Email message with user query sent to back-office failed: "
                    f"{str(e)}."
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
    # Create model
    model = ChatOpenAI(model="gpt-4o")
    logger.info("ChatOpenAI LLM is created.")

    # Create state message for user query
    messages = [HumanMessage(content=user_query)]
    logger.info(f"User query: {user_query}")

    # Initial state for Agent
    initial_state = {
        "messages": messages,
        "user_email_address": None,
        "email_sent": False,
    }

    with SqliteSaver.from_conn_string(":memory:") as memory:
        # Creating the Agent
        abot = Agent(model, [tool], system_prompt=system_prompt, checkpointer=memory)
        logger.info("Agent is created.")

        # Create the Graph stream
        thread = {"configurable": {"thread_id": "1"}}

        # Track the final AImessage
        final_AI_message = None

        # Handle stream processing for graph, log state messages and keep track of
        # final state AImessage
        for event in abot.graph.stream(initial_state, thread):
            for v in event.values():
                for message in v["messages"]:
                    # Log message info for debugging
                    logger.debug(
                        f"Message type: {type(message).__name__}\n"
                        f"Content: {message.content}"
                    )
                    # If there are tool calls, log those separately
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        logger.debug(f"Tool calls: {message.tool_calls}")
                    # Save the last answer to the user from the LLM
                    if message.type == "ai":
                        final_AI_message = message

    if final_AI_message:
        user_answer = final_AI_message.content
    else:
        user_answer = "No answer has been provided to user"
    logger.info(f"Answer to user query: {user_answer}")

    return user_answer


# This section will only run if the script is executed directly
if __name__ == "__main__":
    # Default system prompt
    DEFAULT_PROMPT = """You are a smart research assistant. Use the search engine to look up information. 
    You are allowed to make multiple calls (either together or in sequence). 
    Only look up information when you are sure of what you want. 
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    All questions are about the company AB-hem.
    This step shall be done in this order: 
    1. Always search for an answer on AB Hems homepage only, which is https://ab-hem.se/. 
    2. If no answer is found on AB Hems homepage, then always answer with exactly this message: 
    "Vi kan tyvärr inte svara på din fråga nu."
    """

    # Example of user queries
    # query = "Vad är telefonnumret för felanmälan till AB-hem?"
    # query = "Vem äger AB Hem?"
    query = "Vem är VD på AB Hem?"

    # Trigger the Chatbot Agent flow
    run_agent(DEFAULT_PROMPT, query)
