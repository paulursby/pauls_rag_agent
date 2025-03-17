# Play around with RAG and Agent using Langchain framework

"""Required packages
pip install langchain-community pymupdf
pip install langchain-openai
pip install langchain-text-splitters
pip install chromadb langchain-chroma
"""

import logging
import os

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# from pprint import pprint
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


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
    # list and load all elements of type pdf
    loaded_pages = [
        page.page_content  # Extract text from Document
        for pdf in os.listdir(rag_doc_dir)
        if pdf.endswith(".pdf")
        for page in PyMuPDFLoader(os.path.join(rag_doc_dir, pdf)).load()
    ]

    # Print results
    # print(f"Loaded {len(all_pages)} documents.\n")
    # pprint(all_pages[:])
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

    print(f"\nSplitted data - no of chunks: {len(flatten_chunks)}")
    print(f"Splitted data - type: {type(flatten_chunks)}")
    print(f"Splitted data element - type: {type(flatten_chunks[0])}")
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
    print(f"\nNumber of added data chunks to Chroma vector DB: {len(ids)}")
    return vector_db


## Retriever
def retriever(rag_doc_dir, openai_llm):
    loaded_pages = document_loader(rag_doc_dir)
    chunks = text_splitter(loaded_pages)
    vector_db = vector_database(chunks)

    # Multi-Query Retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(), llm=openai_llm
    )

    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
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
    retriever_qa = retriever(rag_doc_dir, openai_llm)
    chain = create_retrieval_chain(retriever_qa, question_answer_chain)

    response = chain.invoke({"input": query})
    print(f"\nResponse from Q&A chain: {response['answer']}")
    return response


# OpenAI GPT-4o model
openai_llm = create_openai_llm_model()

# Directory containing PDFs
rag_doc_dir = "./RAG_data/CV_Qabot/"

# User query
query = "What does Paul like with Volvo Cars?"
# query = "How many years of experince does Paul have from telecom Industry?"

response = retriever_qa_chain(rag_doc_dir, query, openai_llm)
