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
import tempfile
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

def add_document(file):  
    if file.type == 'application/pdf':
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.read())
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load()
            # st.write(pages)
        return pages   
    else:
        st.warning(f"{file.name} is not a supported file format")

def get_file_name_with_extension(path):
    """Gets the file name with extension from the path."""
    file_name = os.path.basename(path)
    file_name_with_extension = file_name 
    return file_name_with_extension

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
    doc_list=[]
    for file in uploaded_files:
        pages=add_document(file) 
    
        text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200, #striding over the text
        length_function = len)
        texts = text_splitter.split_documents(pages)

        for text in texts:
            doc_list.append(text)
        

    embeddings = OpenAIEmbeddings()
    kbase=FAISS.from_documents(doc_list, embeddings)

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
            result1=rqa1(query)
            st.write('Results generated using GPT4 model:')
            st.write(result1['result'])
            if ((result1['result'])!='No relevant context in documents.'):
                st.write(result1['source_documents'])
                file_name=get_file_name_with_extension(result1['source_documents'][0].metadata['source'])
                st.write('Source: ',file_name)
                page_num=result1['source_documents'][0].metadata['page']
                page_num=page_num+1
                page_num=f"Page#: {page_num:02d}"
                st.write(page_num)  

        if model2:
            result2=rqa2(query)
            st.write('Results generated using GPT3.5 turbo model:')
            st.write(result2['result'])
            if ((result2['result'])!='No relevant context in documents.'):
                st.write(result2['source_documents'])
                file_name=get_file_name_with_extension(result2['source_documents'][0].metadata['source'])
                st.write('Source: ',file_name)
                page_num=result2['source_documents'][0].metadata['page']
                page_num=page_num+1
                page_num=f"Page#: {page_num:02d}"
                st.write(page_num)
        