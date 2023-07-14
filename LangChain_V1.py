import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS,Chroma
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
import os
from langchain.chat_models import ChatOpenAI


os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

def add_document(text,file):  
    if file.type == 'application/pdf':
        doc_reader = PdfReader(file)
        for page in doc_reader.pages:
            text += page.extract_text()
        return text    
    else:
        st.warning(f"{file.name} is not a supported file format")

st.set_page_config(page_title='Contextualized Search for Document Archive using Langchain',layout="wide")
st.write("""
    <style>
        footer {visibility: hidden;}
        body {
            font-family: Arial, sans-serif;
        }
    </style>
""", unsafe_allow_html=True)
st.write(f"<h1 style='font-size: 36px; color: #00555e; font-family: Arial;text-align: center;'>Contextualized Search for Document Archive using LangChain 🦜️🔗</h1>", unsafe_allow_html=True)
# create file uploader
uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True,type=['pdf'])    

if uploaded_files:
    doc_list=""
    text = ""
    for file in uploaded_files:
        dlist=add_document(text,file)
        doc_list+=dlist

    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 100, #striding over the text
    length_function = len)
    texts = text_splitter.split_text(doc_list)

    embeddings = OpenAIEmbeddings()
    kbase=FAISS.from_texts(texts, embeddings)

    prompt_template = """Given the context and the given question,provide a comprehensive answer in detail from the relevant information presented in the context.
           If you don't know the answer, just reply with 'No relevant context in documents', don't try to make up an answer.
            Context: {context}
            Question: {question}
            Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])

    st.write(f"<p style='font-size: 16px; color: red;font-family: Arial;'>Ask a question:</p>",unsafe_allow_html=True)
    query = st.text_input(label='Ask a question:',label_visibility="collapsed")

    retriever = kbase.as_retriever(search_type="similarity", search_kwargs={"k":4})

    rqa1 = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-4",temperature=0),
                                    chain_type="stuff",
                                    retriever=retriever,
                                    chain_type_kwargs={
                                    "prompt": PromptTemplate(template=prompt_template,input_variables=["context", "question"])},
                                    return_source_documents=True)

    rqa2 = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0),
                                    chain_type="stuff",
                                    retriever=retriever,
                                    chain_type_kwargs={
                                    "prompt": PromptTemplate(template=prompt_template,input_variables=["context", "question"])},
                                    return_source_documents=True)    
    
    col1, col2 = st.columns(2,gap="small")
    with st.form("Models"):
        with col1:
            model1 = st.checkbox("GPT4")
        with col2:
            model2 = st.checkbox("GPT3.5-turbo")
        submit_button = st.form_submit_button("Search Document",use_container_width=True)

    if submit_button: 
        if model1:
            result1=rqa1(query)['result']
            st.write('Result generated using GPT4 model')
            st.write(result1)
        if model2:
            result2=rqa2(query)['result']
            st.write('Result generated using GPT3.5-turbo model')
            st.write(result2)
        