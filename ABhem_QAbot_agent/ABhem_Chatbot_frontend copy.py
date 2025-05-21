# Play around with a very basic Gradio GUI,
# mainly used for testing the backend script for AB Hem Q&A bot/chat agent

"""Required packages
pip install gradio
"""

import gradio as gr
from ABhem_Chatbot_backend import process_email, run_agent
from lib.config_loader import Config

# Initialize configuration
config = Config("ABhem_QAbot_agent/config.json")

from lib.logger_setup import get_logger  # noqa: E402 (import not at top of file)

# Setup a logger which can be used by all helper functions
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
    Handle query submission and determine what to show next

    Returns:
    - answer component update
    - email request component update
    - email confirmation component update
    - stored query (str) - to store for email submission
    """
    logger.info("Execute submit_query")
    if not user_query.strip():
        return (
            gr.update(
                value="Oops du glömde fråga nåt! Vänligen försök igen.", visible=True
            ),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
        )

    # Run the agent
    result = run_agent(system_prompt, user_query)

    if result["status"] == "answer_found":
        # Show answer, hide email request
        return (
            gr.update(value=result["content"], visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
        )
    elif result["status"] == "needs_user_email":
        # Hide answer, show email request
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            user_query,
        )
    else:
        # Error occurred, show error message
        return (
            gr.update(value=result["content"], visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
        )


def validate_and_send_email(system_prompt, user_query, user_email):
    """
    Handle email submission after validation

    Returns:
    - answer component update
    - email request component update
    - email confirmation component update
    - email field value update
    """
    logger.info("Execute validate_and_send_email")
    if not user_email.strip():
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=""),
        )

    # Process email submission
    result = process_email(system_prompt, user_query, user_email)

    if result["status"] == "invalid_email":
        # Show validation error, keep email form visible
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=result["content"], visible=True),
            gr.update(value=user_email),
        )
    elif result["status"] == "email_sent":
        # Email sent successfully, show confirmation
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=result["content"], visible=True),
            gr.update(value=""),
        )
    else:
        # Error occurred, show error and keep email form
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=result["content"], visible=True),
            gr.update(value=user_email),
        )


# Create the Gradio interface
with gr.Blocks(title="AB-hem Q&A bot") as demo:
    logger.info("Create the Gradio interface")
    gr.Markdown("# AB-hem Q&A bot")
    gr.Markdown("Fråga på om AB-hem!")

    # Store user query for email submission
    stored_query = gr.State("")

    with gr.Row():
        with gr.Column():  # LEFT COLUMN
            # User query input
            user_query = gr.Textbox(
                label="Din fråga om AB-hem som befintlig eller potentiel kommande hyersgäst:",
                placeholder="Fråga nått om AB-hem...",
                lines=2,
            )

            # Submit button for query
            submit_btn = gr.Button("Skicka fråga")

            # Answer display (conditionally visible)
            answer = gr.Textbox(label="Ditt svar om AB-hem:", lines=5, visible=False)

            # Email collection (conditionally visible)
            with gr.Group(visible=False) as email_group:
                user_email = gr.Textbox(
                    label="Vänligen ange din e-postadress så återkommer vi med svar så snart som möjligt:",
                    placeholder="din.email@exempel.se",
                )
                email_submit_btn = gr.Button("Skicka mail adress")

            # Email confirmation/error message
            email_confirmation = gr.Textbox(label="Status", lines=2, visible=False)

        with gr.Column():  # RIGHT COLUMN
            # System prompt (only visible to admins/testers)
            system_prompt = gr.Textbox(
                label="Enter system prompt:",
                placeholder="You are a smart research assistant...",
                value=DEFAULT_PROMPT,
                lines=5,
            )

    # Set up the query submission action
    submit_btn.click(
        fn=submit_query,
        inputs=[system_prompt, user_query],
        outputs=[
            answer,  # Update answer component
            email_group,  # Update email group visibility
            email_confirmation,  # Update email confirmation
            stored_query,  # Store query for email submission
        ],
    )

    # Set up the email submission action
    email_submit_btn.click(
        fn=validate_and_send_email,
        inputs=[system_prompt, stored_query, user_email],
        outputs=[
            answer,  # Update answer component
            email_group,  # Update email group visibility
            email_confirmation,  # Update email confirmation
            user_email,  # Update email field
        ],
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
