# Play around with RAG and Agent using Langchain framework,
# using my CV and CL as RAG data

"""Required packages
pip install langchain-community pymupdf
pip install chromadb langchain-chroma
pip install langchain langchain-openai
pip install gradio
"""

import logging
import os

import gradio as gr
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# from pprint import pprint
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# from langchain_core.tracers.context import tracing_v2_enabled

# Setting default logging level for modules not having a specified log level defined
logging.basicConfig(level=logging.INFO)
# Configure logging for this module
logger = logging.getLogger("paul_CV_QAbot_logger")
logger.setLevel(logging.DEBUG)


# Content in ~/.bashrc to activate Langsmith tracing
# export LANGSMITH_TRACING=true
# export LANGSMITH_ENDPOINT="https://eu.api.smith.langchain.com"
# export LANGSMITH_API_KEY="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"
# export LANGSMITH_PROJECT="default-pauls-rags"


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
        for page in PyMuPDFLoader(os.path.join(rag_doc_dir, pdf)).load()
    ]

    logger.info(f"Loaded {len(loaded_pages)} documents.")
    return loaded_pages


# Text Splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
    )

    # Chunks will be of type str
    orig_chunks = [text_splitter.split_text(page) for page in data]
    # Flatten the list if you want a single list of all chunks
    flatten_chunks = [chunk for sublist in orig_chunks for chunk in sublist]

    logger.info(f"Splitted data - no of chunks: {len(flatten_chunks)}")
    logger.debug(f"Splitted data - type: {type(flatten_chunks)}")
    logger.debug(f"Splitted data element - type: {type(flatten_chunks[0])}")
    return flatten_chunks


# OpenAI Embedding model
def openai_embedding():
    return OpenAIEmbeddings(model="text-embedding-ada-002")


# Vector store Chroma DB
def vector_database(chunks):
    # Initialize Vector Chroma DB with OpenAI Embeddings
    embedding_model = openai_embedding()
    vector_db = Chroma(
        collection_name="PaulsCVvectorstore",
        embedding_function=embedding_model,
    )

    # Create an ID list that will be used to assign each chunk a known unique identifier
    ids = [str(i) for i in range(0, len(chunks))]
    # Store text chunks in Vector DB
    vector_db.add_texts(chunks, ids=ids)

    logger.info(f"Number of added data chunks to Chroma vector DB: {len(ids)}")
    return vector_db


## Retriever
def retriever_multi_query(rag_doc_dir, openai_llm):
    loaded_pages = document_loader(rag_doc_dir)
    chunks = text_splitter(loaded_pages)
    vector_db = vector_database(chunks)

    # Multi-Query Retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(), llm=openai_llm
    )
    return retriever


# Q&A chain
def retriever_qa_chain(rag_doc_dir, query, openai_llm):
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(openai_llm, prompt)
    retriever_qa = retriever_multi_query(rag_doc_dir, openai_llm)

    # Invoke retriever separately to get the retrieved documents
    retrieved_docs = retriever_qa.invoke(query)
    retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    # Log the retrieved context
    logger.debug(f"Retrieved context for query '{query}':\n{retrieved_context}")

    chain = create_retrieval_chain(retriever_qa, question_answer_chain)

    # A context manager for tracing a specific block of code.
    # with tracing_v2_enabled():
    response = chain.invoke({"input": query})
    logger.info(f"Response from Q&A chain: {response['answer']}")
    return response["answer"]


def retriever_qa_chain_interface(file, query):
    openai_llm = create_openai_llm_model()
    response = retriever_qa_chain(file, query, openai_llm)
    return response


# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa_chain_interface,
    allow_flagging="never",
    inputs=[
        # gr.File(label="Upload PDF File", file_count="single", file_types=[".pdf"], type="filepath"),
        gr.Textbox(
            label="Path to RAG data files", value="./RAG_data/CV_Qabot/", lines=1
        ),  # Default value added
        gr.Textbox(
            label="Input Query", lines=2, placeholder="Type your question here..."
        ),
    ],
    outputs=gr.Textbox(
        label="Output", placeholder="Chatbot answer will appear here..."
    ),
    title="Pauls CV RAG Chatbot",
    description="Add the path to the RAG data file(s) and ask any question. The chatbot will try to answer using the provided document.",
)

# Launch the app
rag_application.launch(server_name="0.0.0.0", server_port=7860)


""" Running application without GUI
# OpenAI GPT-4o model
openai_llm = create_openai_llm_model()

# Directory containing PDFs
rag_doc_dir = "./RAG_data/CV_Qabot/"

# User query
query = "What does Paul like with Volvo Cars?"
# query = "How many years of experience does Paul have from telecom Industry?"

response = retriever_qa_chain(rag_doc_dir, query, openai_llm)
"""
