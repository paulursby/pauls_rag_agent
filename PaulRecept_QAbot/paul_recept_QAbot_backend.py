# Play around with RAG and Agent using Langchain framework,
# using my Recept collection as RAG data

"""Required packages
pip install langchain langchain-openai
pip install langchain_community beautifulsoup4
pip install chromadb langchain-chroma

?pip install langchain-text-splitters
?pip install gradio
"""

import logging
import re

from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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

##############################
# Defining pipeline components
##############################


# Create an OpenAI GPT-4o model
def create_openai_llm_model():
    llm = ChatOpenAI(
        model_name="gpt-4o",  # Use GPT-4o model
        temperature=0.5,  # Controls creativity
        max_tokens=256,  # Limits response length
    )
    return llm


# Document loader
def document_loader(url_list):
    loader = WebBaseLoader(url_list)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents.")

    return docs


# Clean HTML content
def clean_html(html_content):
    """Cleans extracted HTML content by removing scripts, styles, and unnecessary whitespace."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove unwanted elements
    for tag in soup(["script", "style", "meta", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # Extract cleaned text
    clean_text = soup.get_text(separator="\n", strip=True)
    clean_text = re.sub(r"\n+", "\n", clean_text)  # Remove multiple newlines
    clean_text = re.sub(r"\s{2,}", " ", clean_text)  # Remove excessive spaces
    clean_text = clean_text.strip()  # Remove leading/trailing whitespace

    return clean_text


# Clean the extracted content from each doc
def clean_docs(docs):
    cleaned_docs = []
    for i, doc in enumerate(docs):
        logger.info(
            f"Cleaning document {i + 1}/{len(docs)} from {doc.metadata['source']}"
        )
        cleaned_text = clean_html(doc.page_content)
        # `cleaned_docs` contains structured data with cleaned text from each URL
        cleaned_docs.append({"url": doc.metadata["source"], "content": cleaned_text})

        # Logs the cleaned text for debugging
        logger.debug(
            f"Cleaned Document {i + 1} ({doc.metadata['source']}):\n{'=' * 50}\n{cleaned_text[:]}\n{'=' * 50}\n"
        )
    return cleaned_docs


# Text Splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
    )

    # Chunks will be of type str
    orig_chunks = [text_splitter.split_text(page["content"]) for page in data]
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


# Create VectorStoreRetriever
def create_vectore_store_retriever(vector_db):
    # Initialize a retriever from the vector store
    retriever = vector_db.as_retriever(
        search_type="similarity",  # Options: similarity, mmr, similarity_score_threshold
        search_kwargs={
            "k": 3,  # Number of documents to return
            # "score_threshold": 0.7,  # Optional: Only return docs with score above threshold
            # "fetch_k": 10,  # Optional: For MMR, fetch these many documents before reranking
            # "lambda_mult": 0.5,  # Optional: For MMR, diversity factor (0-1)
        },
    )
    logger.info("Retriever successfully created from vector database")
    return retriever


# Search for data in Vector store, using the retriever
def search_vector_store(retriever, query):
    logger.info(f"Searching for: {query}")
    retrieved_docs = retriever.invoke(query)
    logger.info(f"Found {len(retrieved_docs)} relevant documents")

    # For debugging: print the content of retrieved documents
    for i, doc in enumerate(retrieved_docs):
        logger.debug(f"Document {i + 1}:\n{doc.page_content}\n")

    return retrieved_docs


# Search for data in vectore store
def search_data_vector_store(inputs):
    query = inputs["query"]
    retriever = inputs["retriever"]

    # Use the retriever to get documents
    docs = search_vector_store(retriever, query)
    # context = "\n\n".join([doc.page_content for doc in docs])

    return {"context": docs, "input": query}


# Create Prompt
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


####################
# Creating RAG chain
####################

# List of URLs to scrape
urls = [
    "https://vivavinomat.se/recept/italiensk-bongryta/",
    "https://folkofolk.se/vin/recept/grillad-tonfisk-med-mango-och-avokadosalsa",
]

document_loader_runnable = RunnableLambda(document_loader)
clean_docs_runnable = RunnableLambda(clean_docs)
text_splitter_runnable = RunnableLambda(text_splitter)
vector_database_runnable = RunnableLambda(vector_database)

# Build the RAG pipeline
rag_chain = (
    document_loader_runnable
    | clean_docs_runnable
    | text_splitter_runnable
    | vector_database_runnable
)
logger.info("RAG pipeline succesfully created")

vector_db = rag_chain.invoke(urls)
logger.info("RAG pipeline succesfully executed")

####################
# Creating Q&A chain
####################

# Create retriever from the vector store
retriever = create_vectore_store_retriever(vector_db)
logger.info("VectorStoreRetriever successfully created")

search_data_vector_store_runnable = RunnableLambda(search_data_vector_store)

# Build the QA pipeline
qa_chain = (
    search_data_vector_store_runnable
    | prompt
    | create_openai_llm_model()
    | StrOutputParser()
)
logger.info("Q&A pipeline successfully created")

# Execute the Q&A chain
query = "vad ska man servera böngrytan med?"
response = qa_chain.invoke({"query": query, "retriever": retriever})
logger.info(
    f"Q&A pipeline successfully executed with:\n"
    f"User query: {query}\n"
    f"Bot response: {response}"
)
