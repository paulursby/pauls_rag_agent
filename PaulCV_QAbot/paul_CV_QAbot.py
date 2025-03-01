# Play around with RAG and Agent using Langchain framework

"""Required packages
#pip install langchain-community pypdf
pip install langchain-community pymupdf
pip install langchain-openai
pip install langchain-text-splitters
#pip install sentence-transformers
"""

import os

# from langchain_core.runnables import RunnableLambda
# from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from pprint import pprint
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
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

print(f"Splitted text - no of chunks: {len(all_pages_splitted)}")
print(type(all_pages_splitted))
print(type(all_pages_splitted[0]))
# print(f"Splitted text:\n {all_pages_splitted}\n")

###########
# Embedding
###########

# Initialize OpenAI Embeddings
openai_embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

# Query to embed
query = "What does Paul like with Volvo Cars?"

# Get embedding vector for query
embedding_vector_query = openai_embedding.embed_query(query)
print(f"Embedding vector dimension for query: {len(embedding_vector_query)}")

# Get embedding vector for search docs
embedding_vector_docs = openai_embedding.embed_documents(all_pages_splitted)
print(f"No of total Embedding vectors for search docs: {len(embedding_vector_docs)}")
print(f"Embedding vector dimension for search docs: {len(embedding_vector_docs[0])}")
# print(f"Embedding vector example for search docs: {embedding_vector_docs[0]}")
