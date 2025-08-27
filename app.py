from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableSequence , RunnableParallel
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import os
# from langchain.llms import ollama
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
import streamlit as st






def main():
    model = ChatOllama(model="nomic-embed-text:v1.5" , temperature=0)
    st.set_page_config(page_title='Chat with PDF', page_icon='🤖')
    st.header('Chat with your PDF💬...')
    pdf = st.file_uploader('Upload Your PDF here')
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)
        
        
        #Split it into Chunks..
        text_splitter = RecursiveCharacterTextSplitter(
           
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
            
        )
        chunks = text_splitter.split_text(text)
        # st.write(chunks)
        
        #Create embeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
        knowledge_base = FAISS.from_texts(chunks ,embedding=embeddings)
        
    #User chaat here......
    user= st.chat_input('Ask Your Query Here💬 ....')
    if user:
        docs = knowledge_base.similarity_search(user)
        
        llm = ChatOllama(model="gemma3:1b" ,temperature=0.5)
        chain = load_qa_chain(llm , chain_type="stuff")
        response = chain.run(input_documents = docs , question = user)
        
        st.write(response)
        


if __name__ == "__main__":
    main()