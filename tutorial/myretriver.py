from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import streamlit as st

def getDB():
    embeddings = (
        OllamaEmbeddings(model="gemma2:2b")
    )
    db = FAISS.load_local("/Volumes/sunny333/code/genAI/genAI-main/lang1.0/genAI_App/RAG_Ollama/tutorial/vectorDB/index",
                          embeddings,
                          allow_dangerous_deserialization=True)
    
    return db


#---- creating prompt-----

prompt = ChatPromptTemplate.from_template(
    """
    Answer the the following question based on the below context. limit your result t0 100 words:
    <context>
    {context}
    </context>
    
    """
    
)

llm = Ollama(model="gemma2:2b")
doc_chain = create_stuff_documents_chain(llm,prompt)

db = getDB()

retriver = db.as_retriever()

ret_chin = create_retrieval_chain(retriver,doc_chain)

# streamlit code

st.title("basic model testing")
input_text = st.text_input("Ask a question")

if input_text:
    res = ret_chin.invoke({"input": input_text})
    st.write(res["answer"])
   