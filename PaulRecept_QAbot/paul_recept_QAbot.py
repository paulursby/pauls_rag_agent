# Play around with RAG and Agent using Langchain framework,
# using my Recept collection as RAG data

"""Required packages
pip install langchain_community beautifulsoup4
pip install selenium unstructured
?pip install langchain-text-splitters
?pip install chromadb langchain-chroma
?pip install gradio
?pip install langchain langchain-openai
"""

import logging
import os

from langchain_community.document_loaders import SeleniumURLLoader, WebBaseLoader
from langchain_openai import ChatOpenAI

# Setting default logging level for modules not having a specified log level defined
logging.basicConfig(level=logging.INFO)
# Configure logging for this module
logger = logging.getLogger("paul_recept_QAbot_logger")
logger.setLevel(logging.DEBUG)

# Content in ~/.bashrc to activate Langsmith tracing
# export LANGSMITH_TRACING=true
# export LANGSMITH_ENDPOINT="https://eu.api.smith.langchain.com"
# export LANGSMITH_API_KEY="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"
# export LANGSMITH_PROJECT="default-pauls-rags"

# A context manager for tracing a specific block of code.
# with tracing_v2_enabled():


# Create an OpenAI GPT-4o model
def create_openai_llm_model():
    llm = ChatOpenAI(
        model_name="gpt-4o",  # Use GPT-4o model
        temperature=0.5,  # Controls creativity
        max_tokens=256,  # Limits response length
    )
    return llm


# Document loader
def document_loader(rag_doc_dir):
    # List and load all elements of type pdf
    loaded_pages = [
        page.page_content  # Extract text from Document
        for pdf in os.listdir(rag_doc_dir)
        if pdf.endswith(".pdf")
        for page in WebBaseLoader(os.path.join(rag_doc_dir, pdf)).load()
    ]

    logger.info(f"Loaded {len(loaded_pages)} documents.")
    return loaded_pages


# url = "https://recept.se/recept/mexikansk-bongryta-med-majs"
url = "https://vivavinomat.se/recept/italiensk-bongryta/"

# loader = WebBaseLoader(url)
loader = SeleniumURLLoader(urls=[url])

docs = loader.load()

# Print the extracted content
print(docs[0].page_content)
