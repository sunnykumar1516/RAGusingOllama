from dataIngestion import loadPDF

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def create_embeddings():
    doc = loadPDF("data/attention.pdf")
    embeddings = (
        OllamaEmbeddings(model="gemma2:2b")
    )
    vectorStore = FAISS.from_documents(doc, embeddings)
    vectorStore.save_local("vectorDB/index")
    
    
    
create_embeddings()