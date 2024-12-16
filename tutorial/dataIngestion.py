from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def loadPDF(path):
    loader =  PyPDFLoader(path)
    docs  = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    text = text_splitter.split_documents(docs)
    return text




