"""Required packages
pip install gradio
"""

import gradio as gr
from paul_recept_QAbot_backend import (
    create_vectore_store_retriever,
    qa_chain,
    rag_chain,
)


def chatbot_interface(urls, query):
    # Execute the RAG pipeline
    url_list = [url.strip() for url in urls.split(",")]
    vector_db = rag_chain.invoke(url_list)
    retriever = create_vectore_store_retriever(vector_db)
    response = qa_chain.invoke({"query": query, "retriever": retriever})
    return response


with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot for Recipe Queries")
    urls_input = gr.Textbox(label="Enter URLs (comma-separated)")
    query_input = gr.Textbox(label="Enter your question")
    output_text = gr.Textbox(label="Response")
    submit_button = gr.Button("Ask")

    submit_button.click(
        chatbot_interface, inputs=[urls_input, query_input], outputs=output_text
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
