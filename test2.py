import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

st.set_page_config(page_title="RAG Q&A", layout="wide")
st.title("RAG Document Q&A With Groq and Llama3")
st.markdown("Welcome! Please enter your Groq API key below to begin.")

if "groq_api_key" not in st.session_state not in st.session_state:
    with st.form("api_key_form"):
        groq_api_key = st.text_input("Enter your GROQ API Key", type="password")
        submitted = st.form_submit_button("Submit")

    if not submitted or not groq_api_key:
        st.warning("Please submit both API keys to proceed.")
        st.stop()
    else:
        try:
            _ = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            st.error(f"Invalid Hugging Face configuration: {e}")
            st.stop()

        try:
            _ = ChatGroq(model_name="Llama3-8b-8192", api_key=groq_api_key)
        except Exception as e:
            st.error(f"Invalid GROQ API Key: {e}")
            st.stop()

        st.session_state.groq_api_key = groq_api_key
       

groq_api_key = st.session_state.groq_api_key

# Step 1: Load and split PDFs
if st.button("Load and Split PDFs"):
    try:
        loader = PyPDFDirectoryLoader("research_paps")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        st.session_state.docs = split_docs
        st.success(f"Loaded and split {len(split_docs)} document chunks.")
    except Exception as e:
        st.error(f" Failed to load/split PDFs: {e}")

# Step 2: Create vector database
if st.button("Create Vector DB"):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        db = FAISS.from_documents(st.session_state.docs, embeddings)
        st.session_state.vectors = db
        st.success("Vector DB created.")
    except Exception as e:
        st.error(f"Failed to create vector DB: {e}")

# Step 3: Prompt input
user_prompt = st.text_input("Ask a question about the documents")

# Step 4: Query and respond
if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please load and embed documents first.")
    else:
        try:
            prompt = ChatPromptTemplate.from_template("""
<context>
{context}
</context>
Question: {input}
""")

            llm = ChatGroq(model_name="Llama3-8b-8192", api_key=groq_api_key)
            retriever = st.session_state.vectors.as_retriever()
            doc_chain = create_stuff_documents_chain(llm, prompt)
            chain = create_retrieval_chain(retriever, doc_chain)

            response = chain.invoke({'input': user_prompt})
            st.subheader("Answer")
            st.write(response['answer'])
        except Exception as e:
            st.error(f"Failed to generate answer: {e}")