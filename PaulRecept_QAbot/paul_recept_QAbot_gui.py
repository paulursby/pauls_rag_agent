# Play around with RAG and Agent using Langchain framework invl LCEL,
# using some of my Recept collection as RAG data
# This is GUI of the Q&Abot

"""Required packages
pip install gradio
"""

import logging

import gradio as gr
from paul_recept_QAbot_backend import (
    create_vectore_store_retriever,
    qa_chain,
    rag_chain,
)

# Setting default logging level for modules not having a specified log level defined
logging.basicConfig(level=logging.INFO)
# Configure logging for this module
logger = logging.getLogger("paul_recept_QAbot_gui_logger")
logger.setLevel(logging.DEBUG)


def chatbot_interface(urls, query):
    logger.info("Question submitted from GUI")

    url_list = [url.strip() for url in urls.split(",")]
    vector_db = rag_chain.invoke(url_list)
    logger.info("RAG pipeline succesfully executed")

    retriever = create_vectore_store_retriever(vector_db)
    logger.info("VectorStoreRetriever successfully created")

    response = qa_chain.invoke({"query": query, "retriever": retriever})
    logger.info(
        f"Q&A pipeline successfully executed with:\n"
        f"User query: {query}\n"
        f"Bot response: {response}"
    )

    return response


with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot for Recipe Queries")
    urls_input = gr.Textbox(label="Enter URLs (comma-separated)")
    query_input = gr.Textbox(label="Enter your question")
    output_text = gr.Textbox(label="Response")
    submit_button = gr.Button("Ask")
    logger.info("GUI successfully started")

    submit_button.click(
        chatbot_interface, inputs=[urls_input, query_input], outputs=output_text
    )


demo.launch(server_name="0.0.0.0", server_port=7860)
