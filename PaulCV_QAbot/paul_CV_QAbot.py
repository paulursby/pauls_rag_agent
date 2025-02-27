# Play around with RAG and Agent using Langchain framework

"""Required packages
pip install langchain-community pypdf
pip install langchain-community pymupdf
"""

from pprint import pprint

# from pathlib import Path
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader

# loader = PyPDFLoader(file_path="./RAG_data/CV General Paul C.pdf")
# pages = loader.load_and_split()

loader = PyMuPDFLoader(file_path="./RAG_data/CV General Paul C.pdf")
pages = loader.load()

pprint(pages[0])
