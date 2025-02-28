# Play around with RAG and Agent using Langchain framework

"""Required packages
pip install langchain-community pypdf
pip install langchain-community pymupdf
"""

import os
from pprint import pprint

# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader

# Load Documents
################

# loader = PyPDFLoader(file_path="./RAG_data/CV General Paul C.pdf")
# pages = loader.load_and_split()

# loader = PyMuPDFLoader(file_path="./RAG_data/CV General Paul C.pdf")
# pages = loader.load()

# Directory containing PDFs
pdf_directory = "./RAG_data/"

# Load all PDFs
all_pages = [
    page for pdf in os.listdir(pdf_directory) if pdf.endswith(".pdf")
    for page in PyMuPDFLoader(os.path.join(pdf_directory, pdf)).load()
]

# Print results
print(f"Loaded {len(all_pages)} documents.\n")
pprint(all_pages[:])
