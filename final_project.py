# from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyMuPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings

# Task 1
"""
pdf_url = "./RAG_data/A-Comprehensive-Review-of-Low-Rank-Adaptation-in-Large-Language-Models-for-Efficient-Parameter-Tuning-1.pdf"

def document_loader(file):
    loader = PyMuPDFLoader(file)
    data = loader.load()
    return data

data = document_loader(pdf_url)
full_text = " ".join(doc.page_content for doc in data)
print(full_text[:1000])
"""

# Task 2

latex_text = """
    \documentclass{article}
    
    \begin{document}
    
    \maketitle
    
    \section{Introduction}
    Large language models (LLMs) are a type of machine learning model that can be trained on vast amounts of text data to generate human-like language. In recent years, LLMs have made significant advances in a variety of natural language processing tasks, including language translation, text generation, and sentiment analysis.
    
    \subsection{History of LLMs}
    The earliest LLMs were developed in the 1980s and 1990s, but they were limited by the amount of data that could be processed and the computational power available at the time. In the past decade, however, advances in hardware and software have made it possible to train LLMs on massive datasets, leading to significant improvements in performance.
    
    \subsection{Applications of LLMs}
    LLMs have many applications in industry, including chatbots, content creation, and virtual assistants. They can also be used in academia for research in linguistics, psychology, and computational linguistics.
    
    \end{document}
"""
"""
latex_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.LATEX, chunk_size=60, chunk_overlap=0
)
latex_docs = latex_splitter.create_documents([latex_text])
print(latex_docs)
"""

# Task 3

# Run cell in jupiter lab and took screen dumps from there.


# Task 4

loader = TextLoader("./RAG_data/new-Policies.txt")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)
chunks = text_splitter.split_documents(data)

# Initialize OpenAI Embeddings
openai_embedding = OpenAIEmbeddings(model="text-embedding-ada-002")


ids = [str(i) for i in range(0, len(chunks))]

# Create Vector Chroma DB with OpenAI Embeddings
vectordb = Chroma.from_documents(chunks, openai_embedding, ids=ids)

for i in range(3):
    print(vectordb._collection.get(ids=str(i)))

# Similarity serach
query = "Smoking policy"
docs = vectordb.similarity_search(query, k=5)
print(f"Similarity serach result: {docs}")


# Task 5

"""
vectordb = Chroma.from_documents(documents=chunks, embedding=openai_embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

query = "Email policy"
docs = retriever.invoke(query)
print(f"Retriever search result: {docs}")
"""

# Task 6

# Run cell in jupiter lab and took screen dumps from there.
