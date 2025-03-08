# Play around with RAG and Agent using Langchain framework

"""Required packages
#pip install langchain-community pypdf
pip install langchain-community pymupdf
pip install langchain-openai
pip install langchain-text-splitters
pip install chromadb langchain-chroma
pip install langchain_community faiss-cpu
#pip install sentence-transformers
"""

import logging
import os

# from langchain_core.runnables import RunnableLambda
# from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# from pprint import pprint
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader

# from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

################
# Load Documents
################

# loader = PyPDFLoader(file_path="./RAG_data/CV General Paul C.pdf")
# pages = loader.load_and_split()

# loader = PyMuPDFLoader(file_path="./RAG_data/CV General Paul C.pdf")
# pages = loader.load()

# Directory containing PDFs
pdf_directory = "./RAG_data/"

# Load all PDFs
"""
# list elements are of type Document
all_pages = [
    page for pdf in os.listdir(pdf_directory) if pdf.endswith(".pdf")
    for page in PyMuPDFLoader(os.path.join(pdf_directory, pdf)).load()
]
"""
# list elements are of type str
all_pages = [
    page.page_content  # Extract text from Document
    for pdf in os.listdir(pdf_directory)
    if pdf.endswith(".pdf")
    for page in PyMuPDFLoader(os.path.join(pdf_directory, pdf)).load()
]

# Print results
# print(f"Loaded {len(all_pages)} documents.\n")
# pprint(all_pages[:])

################################
# Put Whole Document into Prompt
################################


# Function to create an OpenAI GPT-4o model
def llm_model():
    llm = ChatOpenAI(
        model_name="gpt-4o",  # Use GPT-4o model
        temperature=0.5,  # Controls creativity
        max_tokens=256,  # Limits response length
    )
    return llm


# Initialize GPT-4o
openai_llm = llm_model()

# response = openai_llm.invoke("How are you?")
# print(response.content)

# Create prompt template
template = """According to the document content here 
            {content},
            answer this question 
            {question}.
            Do not try to make up the answer.
                
            YOUR RESPONSE:
"""
prompt_template = PromptTemplate(
    template=template, input_variables=["content", "question"]
)

# Create a runnable chain
# query_chain = prompt_template | openai_llm | RunnableLambda(lambda x: x.content)

# Example input
data = {"content": all_pages, "question": "What does Paul like with Volvo Cars?"}

# Invoke the chain
# response = query_chain.invoke(data)

# Print the response
# print(response)


################
# Text Splitting
################

"""
text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)
"""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)

# print(f"Unsplitted text:\n {all_pages[0]}\n")
# print(type(all_pages))
# print(type(all_pages[0]))

# Chunks are of type str
all_pages_splitted = [text_splitter.split_text(page) for page in all_pages]
# Chunks of type Document
# all_pages_splitted = [text_splitter.create_documents([page], metadatas=[{"document":"Pauls CV in pdf"}]) for page in all_pages]

# Flatten the list if you want a single list of all chunks
all_pages_splitted = [chunk for sublist in all_pages_splitted for chunk in sublist]

print(f"\nSplitted text - no of chunks: {len(all_pages_splitted)}")
print(f"Splitted text type: {type(all_pages_splitted)}")
print(f"Splitted text element type: {type(all_pages_splitted[0])}")
# print(f"Splitted text:\n {all_pages_splitted}\n")

########################
# Embedding using OpenAI
########################

# Initialize OpenAI Embeddings
openai_embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

# Query to embed
# query = "What does Paul like with Volvo Cars?"

# Get embedding vector for query
# embedding_vector_query = openai_embedding.embed_query(query)
# print(f"Embedding vector dimension for query: {len(embedding_vector_query)}")

# Get embedding vector for search docs
# embedding_vector_docs = openai_embedding.embed_documents(all_pages_splitted)
# print(f"No of total Embedding vectors for search docs: {len(embedding_vector_docs)}")
# print(f"Embedding vector dimension for search docs: {len(embedding_vector_docs[0])}")
# print(f"Embedding vector example for search docs: {embedding_vector_docs[0]}")

########################
# Vector store Chroma DB
########################

# Initialize Vector Chroma DB with OpenAI Embeddings
vector_store_chroma = Chroma(
    collection_name="PaulsCVvectorstore",
    embedding_function=openai_embedding,
)

# Create an ID list that will be used to assign each chunk a known unique identifier
ids = [str(i) for i in range(0, len(all_pages_splitted))]

# Store text chunks in Vector DB
vector_store_chroma.add_texts(all_pages_splitted, ids=ids)
print(f"\nNumber of added text chunks to Chroma vector DB: {len(ids)}")
"""
print(
    f"\nText chunk example(s) from Chroma vector DB: {vector_store_chroma.get(ids=['0'])}"
)
"""

"""
# Query the Vector Store
query = "What does Paul like with Volvo Cars?"
retrieved_docs = vector_store_chroma.similarity_search(
    query, k=3
)  # Retrieve top 3 matches

# Print Retrieved Results
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\nSimilarity match {i}: {doc.page_content}")
"""

########################
# Vector store FAISS DB
########################
"""
# Create an ID list that will be used to assign each chunk a known unique identifier
ids = [str(i) for i in range(0, len(all_pages_splitted))]

# Initialize Vector FAISS DB with OpenAI Embeddings and store texts chunks
vector_store_faiss = FAISS.from_texts(all_pages_splitted, openai_embedding, ids=ids)

# Store text chunks in Vector DB
# vector_store.add_texts(all_pages_splitted, ids=ids)
print(f"\nNumber of added text chunks to FAISS vector DB: {len(ids)}")
print(
    f"\nText chunk example(s) from FAISS vector DB: {vector_store_faiss.get_by_ids(['0'])}"
)

# Query the Vector Store
query = "What does Paul like with Volvo Cars?"
retrieved_docs = vector_store_faiss.similarity_search(
    query, k=3
)  # Retrieve top 3 matches

# Print Retrieved Results
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\nSimilarity match {i}: {doc.page_content}")
"""

###########
# Retriever
###########

# query = "What does Paul like with Volvo Cars?"
query = "How many years of experince does Paul have from telecom Industry?"

# Distance-based vector database retrieval
"""
retriever = vector_store_chroma.as_retriever(
    search_type="similarity",  # similarity(default), mmr or similarity_score_threshold
    # search_kwargs={"k": 2, "score_threshold": 0.4},  # Default=4
)
"""

# Multi-Query Retriever
retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store_chroma.as_retriever(), llm=openai_llm
)
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Self-Querying Retriever
# TBD

# Parent Document Retriever
# TBD

retrieved_docs = retriever.invoke(query)

# Print Retrieved Results
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\nSimilarity match {i}: {doc.page_content}")


###########
# Q&A chain
###########

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
chain = create_retrieval_chain(retriever, question_answer_chain)

response = chain.invoke({"input": query})
print(f"\nResponse from Q&A chain: {response['answer']}")
