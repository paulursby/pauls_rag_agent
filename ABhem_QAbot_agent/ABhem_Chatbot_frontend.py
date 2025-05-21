"""
Frontend script using enhanced Gradio GUI with proper checkpointing support for
resumable conversation flow

Precondition is that this package is pip installed:
pip install gradio
"""

import gradio as gr
from ABhem_Chatbot_backend import run_agent_with_checkpoint
from lib.config_loader import Config

# Initialize configuration
config = Config("ABhem_QAbot_agent/config.json")

from lib.logger_setup import get_logger  # noqa: E402

# Setup logging facility
logger = get_logger("ABhem_Chatbot_frontend", config)
logger.info("Logging facility is setup.")

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


def submit_query(system_prompt, user_query):
    """
    Handle initial query submission using the enhanced checkpointing system.
    Each submission starts a fresh conversation.

    Args:
        system_prompt (str): The system prompt for the agent
        user_query (str): The user's question

    Returns:
        Tuple of UI component updates and state variables
    """
    logger.info("Execute submit_query with checkpointing.")

    # Clean user_query
    user_query = user_query.strip()

    # If empty user query is entered
    if not user_query:
        logger.info("Empty user query is entered, so a retry is required.")
        return (
            gr.update(
                value="Oops du glömde fråga nåt! Vänligen försök igen.", visible=True
            ),
            gr.update(visible=False),
            gr.update(visible=False),
            "",  # stored_query
            "",  # thread_id
        )

    # Always start a new conversation (no thread_id passed)
    result = run_agent_with_checkpoint(
        system_prompt=system_prompt, user_query=user_query, thread_id=None
    )

    thread_id = result.get("thread_id", "")

    if result["status"] == "answer_found":
        # Show answer, hide email request
        logger.info("Answer found, displaying result.")
        return (
            gr.update(value=result["content"], visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "",  # stored_query (not needed)
            thread_id,
        )
    elif result["status"] == "needs_email":
        # Hide answer, show email request
        logger.info("No answer found, requesting user's email address.")
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            user_query,  # stored_query
            thread_id,
        )
    else:
        # Error occurred, show error message
        logger.error(f"Error in query processing: {result['status']}.")
        return (
            gr.update(value=result["content"], visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "",  # stored_query (not needed)
            thread_id,
        )


def validate_and_send_email(system_prompt, user_query, user_email, thread_id):
    """
    Handle email submission using the enhanced checkpointing system.

    Args:
        system_prompt (str): The system prompt for the agent
        user_query (str): The user's original question
        user_email (str): The user's email address
        thread_id (str): Thread ID from the previous conversation

    Returns:
        Tuple of UI component updates
    """
    logger.info("Execute validate_and_send_email with checkpointing.")

    # If empty user email address is entered
    if not user_email.strip():
        logger.warning("Empty email address provided.")
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(
                value="Oops du glömde ange din ange din e-postadress! Vänligen försök igen.",
                visible=True,
            ),
            gr.update(value=""),
        )

    # Use the enhanced backend function to resume the conversation with email
    result = run_agent_with_checkpoint(
        system_prompt=system_prompt,
        user_query=user_query,
        thread_id=thread_id,
        user_email=user_email,
    )

    if result["status"] == "invalid_email":
        # Show backend validation error, keep email form visible
        logger.warning(f"Invalid email provided: {user_email}.")
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=result["content"], visible=True),
            gr.update(
                value=user_email
            ),  # Keep the entered email so user can correct it
        )

    elif result["status"] == "email_sent":
        # Email sent successfully, show confirmation
        logger.info("Email sent successfully to back office.")
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=result["content"], visible=True),
            gr.update(value=""),
        )
    elif result["status"] == "email_failed":
        # Sending email failed, show error and keep email form
        logger.error("Sending email failed.")
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=result["content"], visible=True),
            gr.update(value=user_email),
        )
    else:
        # Other error occurred, show error message
        logger.error(f"Error in query processing: {result['status']}.")
        return (
            gr.update(value=result["content"], visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=""),
        )


def reset_conversation():
    """
    Reset the conversation to start fresh.

    Returns:
        Tuple of UI component updates to reset the interface
    """
    logger.info("Resetting conversation.")
    return (
        gr.update(value="", visible=False),  # answer
        gr.update(visible=False),  # email_group
        gr.update(value="", visible=False),  # email_confirmation
        gr.update(value=""),  # user_email
        "",  # stored_query
        "",  # thread_id
    )


# Create the enhanced Gradio interface
with gr.Blocks(title="AB-hem Q&A bot") as demo:
    logger.info("Create the enhanced Gradio interface.")
    gr.Markdown("# AB-hem Q&A bot")
    gr.Markdown("Fråga på om AB-hem!")

    # State variables for conversation management
    stored_query = gr.State("")
    thread_id = gr.State("")

    with gr.Row():
        with gr.Column():  # LEFT COLUMN - User Interface
            # User query input
            user_query = gr.Textbox(
                label="Din fråga om AB-hem som befintlig eller potentiell kommande hyresgäst:",
                placeholder="Fråga något om AB-hem...",
                lines=2,
            )

            # Submit button for query
            submit_btn = gr.Button("Skicka fråga")

            # Answer display (conditionally visible)
            answer = gr.Textbox(
                label="Ditt svar om AB-hem:", lines=5, visible=False, interactive=False
            )

            # Email collection (conditionally visible)
            with gr.Group(visible=False) as email_group:
                user_email = gr.Textbox(
                    label="Vi hittar inte svaret på din fråga just nu, men ange din "
                    "e-postadress så återkommer vi med svar så snart som möjligt:",
                    placeholder="din.email@exempel.se",
                )
                email_submit_btn = gr.Button("Skicka e-postadress")

            # Email confirmation/error message
            email_confirmation = gr.Textbox(
                label="Status", lines=2, visible=False, interactive=False
            )

        with gr.Column():  # RIGHT COLUMN - Admin/Testing Interface
            gr.Markdown("### Admin/Testing Interface")

            # System prompt (visible to admins/testers)
            system_prompt = gr.Textbox(
                label="System prompt:",
                placeholder="You are a smart research assistant...",
                value=DEFAULT_PROMPT,
                lines=8,
            )

    # Set up event handlers

    # Query submission with automatic reset
    submit_btn.click(
        fn=reset_conversation,  # First reset the conversation
        inputs=[],
        outputs=[
            answer,
            email_group,
            email_confirmation,
            user_email,
            stored_query,
            thread_id,
        ],
    ).then(
        fn=submit_query,  # Then submit the new query
        inputs=[system_prompt, user_query],
        outputs=[
            answer,
            email_group,
            email_confirmation,
            stored_query,
            thread_id,
        ],
    ).then(
        fn=lambda: "",  # Clear the user query input
        inputs=[],
        outputs=[user_query],
    )

    # Email submission
    email_submit_btn.click(
        fn=validate_and_send_email,
        inputs=[system_prompt, stored_query, user_email, thread_id],
        outputs=[
            answer,
            email_group,
            email_confirmation,
            user_email,
        ],
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want to create a public link
        debug=True,  # Enable debug mode for development
    )
