"""
This is the backend module of the AB Hem Chat bot agent, using e.g ChatOpenAI,
LangChain, LangGraph and TavilySearch capabilities.

Precondition is that these packages are pip installed:
pip install langgraph langchain-core langchain-openai langchain-community
tavily-python langgraph-checkpoint-sqlite
"""

import operator
import smtplib
import ssl
import uuid
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
from lib.config_loader import Config
from lib.helper_functions import is_valid_email
from lib.logger_setup import get_logger, setup_logging

# Initialize configuration
config = Config("ABhem_QAbot_agent/config.json")

# Register secure configuration parameter, by mapping parameters to env variables:
Config.register_secure_param("backend.back_office_email_sender_pwd", "EMAIL_SENDER_PWD")
Config.register_secure_param("backend.open_api_key", "OPENAI_API_KEY")
Config.register_secure_param("backend.tavily_api_key", "TAVILY_API_KEY")

# Setup logging facility
setup_logging(config)
logger = get_logger("ABhem_Chatbot_backend", config)
logger.info("Logging facility is setup.")

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
tavily_api_key = config.get_param("backend", "tavily_api_key", default="")
logger.info("Config data is fetched to define Tavily search tool.")

tool = TavilySearchResults(
    api_key=tavily_api_key,
    max_results=10,
    include_domains=["ab-hem.se"],
    search_depth="advanced",
    # TODO: Shall this be tried out?
    # include_raw_content=True,
)
logger.info("Tavily search tool is defined.")


# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_email_address: Optional[str]
    email_sent: Optional[bool]
    session_id: Optional[str]


# Define the Agent Graph
class Agent:
    """
    A conversational agent that processes user queries using a state graph workflow,
    with proper checkpointing for resumable email collection.

    The Agent uses a state graph to manage conversation flow, leveraging an LLM model
    for generating responses and tools for performing actions. It can route conversations
    to different paths based on the content of messages, collect email addresses, and
    forward unresolved queries to backend support staff.
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
        graph.add_node("await_email", self.await_email)
        graph.add_node("send_email", self.send_email)

        # Add edges
        graph.add_conditional_edges(
            "llm",
            self.route_next_step,
            {
                "action": "action",
                "await_email": "await_email",
                "end": END,
            },
        )
        graph.add_edge("action", "llm")

        # Add conditional edge to handle pause/resume
        graph.add_conditional_edges(
            "await_email",
            self.check_email_status,
            {
                "wait_for_email": END,  # Pause here when no email
                "proceed_to_send": "send_email",  # Continue when email provided
            },
        )
        graph.add_edge("send_email", END)

        graph.set_entry_point("llm")
        # Compile with interrupt capability
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_after=["await_email"],  # Interrupt after await_email executes
        )

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
        - Await/collect user email address for unresolved queries
        - Send mail to backoffice for unresolved queries
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

        # If no answer is found, next step is await email collection
        if "vi kan tyvärr inte svara på din fråga nu" in content:
            logger.info("Next step is to await email address collection.")
            return "await_email"

        # Otherwise an answer is found, next step is the END
        logger.info("Next step is to exit the Agent flow, since an answer is found.")
        return "end"

    def check_email_status(self, state: AgentState) -> str:
        """
        Check if email has been collected to determine next step.
        This runs after await_email and decides whether to pause or continue.
        """
        if state.get("user_email_address"):
            logger.info("Email address found in state, proceeding to send email")
            return "proceed_to_send"
        else:
            logger.info("No email address, pausing execution")
            return "wait_for_email"

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

    def await_email(self, state: AgentState):
        """
        Pause execution and signal that user email address collection is needed.
        This creates a checkpoint where execution can be resumed.

        Args:
            state (AgentState): The current state containing messages.

        Returns:
            dict: Dictionary containing a system event message indicating the need for
            an email.
        """
        # Create a system event message indicating email collection is needed
        # TODO: change content to: Awaiting user email address and type of message
        email_request = ChatMessage(
            role="system",
            content="awaiting_email",
        )
        logger.info("Pausing execution to await user email address collection.")

        return {"messages": [email_request]}

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
        # Fetch config data
        back_office_email_receiver = config.get_param(
            "backend", "back_office_email_receiver", default=""
        )
        back_office_email_sender = config.get_param(
            "backend", "back_office_email_sender", default=""
        )
        back_office_email_sender_pwd = config.get_param(
            "backend", "back_office_email_sender_pwd", default=""
        )
        logger.info("Config data is fetched for sending email.")

        # Get user email from state
        user_email_address = state.get("user_email_address", "")

        # Extract the original user query, which is the first HumanMessage in the
        # messages list from state
        user_query = "No query found"
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                user_query = message.content
                break

        # Create email
        email = EmailMessage()
        email["From"] = back_office_email_sender
        email["To"] = back_office_email_receiver
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
                server.login(back_office_email_sender, back_office_email_sender_pwd)
                server.send_message(email)

                # Add a state message that email is succesfully sent
                # TODO: Update content to more info see old file?
                confirmation = ChatMessage(
                    role="system",
                    content="email_sent_success",
                )
                logger.info(
                    "Email message with user query is succesfully sent to back-office."
                )

                return {"messages": [confirmation], "email_sent": True}

            except Exception as e:
                # Add a state message that sent email failed
                # TODO: Update content to more info see old file?
                confirmation = ChatMessage(
                    role="system",
                    content="email_sent_failed",
                )
                logger.error(
                    "Email message with user query sent to back-office failed: "
                    f"{str(e)}."
                )

                return {"messages": [confirmation], "email_sent": False}


# Create persistent checkpointer
def get_checkpointer():
    """Get a persistent SQLite checkpointer for state management."""
    logger.info("Get a persistent SQLite checkpointer for state management.")
    return SqliteSaver.from_conn_string("agent_checkpoints.db")


# Function to run the agent and return the answer or status
def run_agent_with_checkpoint(
    system_prompt, user_query, thread_id=None, user_email=None
):
    """
    Run the agent with proper checkpointing support for resumable execution.

    Args:
        system_prompt (str): The system prompt for the assistant
        user_query (str): The user's question
        thread_id (str, optional): Thread ID to resume existing conversation
        user_email (str, optional): User's email address if collecting

    Returns:
        dict: A dictionary containing:
            - 'status': str - 'answer_found', 'needs_email', 'email_sent',
            'email_failed', or 'error'
            - 'content': str - The response content
            - 'thread_id': str - Thread ID for conversation continuity
    """
    # Generate thread_id if not provided (new conversation)
    if not thread_id:
        thread_id = str(uuid.uuid4())
        logger.info(f"Generate new thread_id: {thread_id}")
    else:
        logger.info(f"Resume conversation with thread_id: {thread_id}")

    # Create thread configuration/graph stream
    thread_config = {"configurable": {"thread_id": thread_id}}

    # Validate email if provided
    if user_email:
        is_valid, validation_message = is_valid_email(user_email.strip())
        if not is_valid:
            logger.warning(
                f"Invalid email address provided: {user_email} - {validation_message}"
            )
            return {
                "status": "invalid_email",
                "content": f"Ogiltig e-postadress: {validation_message}. Vänligen ange en giltig e-postadress.",
                "thread_id": thread_id,
            }
        else:
            user_email = validation_message
            logger.info(f"Email validated and normalized: {user_email}")

    # Create model
    open_api_key = config.get_param("backend", "open_api_key", default="")
    logger.info("Config data is fetched to create ChatOpenAI LLM.")

    model = ChatOpenAI(model="gpt-4o", api_key=open_api_key)
    logger.info("ChatOpenAI LLM is created.")

    try:
        with get_checkpointer() as checkpointer:
            # Create the Agent
            abot = Agent(
                model, [tool], system_prompt=system_prompt, checkpointer=checkpointer
            )
            logger.info("Agent is created with persistent checkpointer.")

            # Check if this is a resume operation with email
            if thread_id and user_email:
                logger.info("Resuming conversation with email address provided.")

                # First, update the checkpoint state with the email address
                abot.graph.update_state(
                    thread_config, {"user_email_address": user_email}
                )
                logger.info("Updated checkpoint state with user email address.")

                # Resume graph execution from where it paused
                # Use stream with None to continue from the current checkpoint
                for event in abot.graph.stream(None, thread_config):
                    # Below is just for logging purpose
                    for v in event.values():
                        for message in v["messages"]:
                            # Log message info for debugging
                            logger.debug(
                                f"Message type: {type(message).__name__}. "
                                f"Content: {message.content}"
                            )
                            # If there are tool calls, log those separately
                            if hasattr(message, "tool_calls") and message.tool_calls:
                                logger.debug(f"Tool calls: {message.tool_calls}")

            else:
                # New conversation
                logger.info(f"Start new conversation with query: {user_query}")
                initial_state = {
                    "messages": [HumanMessage(content=user_query)],
                    "user_email_address": None,
                    "email_sent": False,
                    "session_id": thread_id,
                }

                # Execute the graph
                for event in abot.graph.stream(initial_state, thread_config):
                    pass
                    # TODO: Below is not working when requesting email
                    # Below is just for logging purpose
                    """
                    for v in event.values():
                        for message in v["messages"]:
                            # Log message info for debugging
                            logger.debug(
                                f"Message type: {type(message).__name__}. "
                                f"Content: {message.content}"
                            )
                            # If there are tool calls, log those separately
                            if hasattr(message, "tool_calls") and message.tool_calls:
                                logger.debug(f"Tool calls: {message.tool_calls}")
                    # """

            # Track execution state
            final_ai_message = None
            awaiting_email = False
            email_status = None

            # Get final state to determine response
            final_state = abot.graph.get_state(thread_config)

            # Analyze final state to determine response
            if final_state and final_state.values:
                state_values = final_state.values

                # Check if email was succesfully sent or failed
                if state_values.get("email_sent"):
                    email_status = "sent"
                elif state_values.get("user_email_address") and not state_values.get(
                    "email_sent"
                ):
                    email_status = "failed"

                # Check for awaiting email state and final message on user query
                for message in state_values.get("messages", []):
                    if hasattr(message, "role") and message.role == "system":
                        if message.content == "awaiting_email":
                            awaiting_email = True
                    elif message.type == "ai":
                        final_ai_message = message

            # Determine response based on final state
            if awaiting_email and not user_email:
                return {
                    "status": "needs_email",
                    "content": "Vi kan tyvärr inte svara på din fråga nu.",
                    "thread_id": thread_id,
                }
            elif email_status == "sent":
                return {
                    "status": "email_sent",
                    "content": "Tack! Vi kommer att skicka svar till din e-postadress så snart som möjligt.",
                    "thread_id": thread_id,
                }
            elif email_status == "failed":
                return {
                    "status": "email_failed",
                    "content": "Det gick inte att skicka din förfrågan. Vänligen försök igen senare.",
                    "thread_id": thread_id,
                }
            elif final_ai_message:
                return {
                    "status": "answer_found",
                    "content": final_ai_message.content,
                    "thread_id": thread_id,
                }
            else:
                return {
                    "status": "error",
                    "content": "Ett fel uppstod. Vänligen försök igen.",
                    "thread_id": thread_id,
                }

    except Exception as e:
        logger.error(f"Error in run_agent_with_checkpoint: {str(e)}")
        return {
            "status": "error",
            "content": "Ett fel uppstod. Vänligen försök igen.",
            "thread_id": thread_id,
        }
