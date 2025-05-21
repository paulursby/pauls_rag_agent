# Play around with a very basic Gradio GUI,
# mainly used for testing the backend script for AB Hem Q&A bot/chat agent

"""Required packages
pip install gradio
"""

import gradio as gr
from ABhem_QAbot_backend import run_agent

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

# Create the Gradio interface
with gr.Blocks(title="AB-hem Q&A bot") as demo:
    gr.Markdown("# AB-hem Q&A bot")
    gr.Markdown("Fråga på om AB-hem!")

    with gr.Row():
        with gr.Column():
            # Input components
            system_prompt = gr.Textbox(
                label="Enter system prompt:",
                placeholder="You are a smart research assistant...",
                value=DEFAULT_PROMPT,
                lines=5,
            )

            user_query = gr.Textbox(
                label="Din fråga om AB-hem som befintlig eller potentiel kommande hyesgäst:",
                placeholder="Fråga nått om AB-hem...",
                lines=2,
            )

            submit_btn = gr.Button("Submit")

        with gr.Column():
            # Output component
            answer = gr.Textbox(label="Ditt svar om AB-hem:", lines=10)

    # Set up the submission action
    submit_btn.click(fn=run_agent, inputs=[system_prompt, user_query], outputs=answer)

# Launch the app
demo.launch(server_name="0.0.0.0", server_port=7860)
