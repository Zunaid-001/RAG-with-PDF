import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv(dotenv_path="D:/python with genAI/langchain/.env")
google_key = os.getenv("GOOGLE_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if google_key:
    os.environ['GOOGLE_API_KEY'] = google_key
if groq_key:
    os.environ['GROQ_API_KEY'] = groq_key
    
# Initialize session state keys to avoid AttributeError
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "docs" not in st.session_state:
    st.session_state.docs = None

# Groq LLM (note: model name must be lowercase)
llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.3-70b-versatile")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Vector embedding creation
def create_vector_embedding():
    if st.session_state.vectors is None:
        # Correct model name: "embedding-001" (no "models/")sentence
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit UI
st.title("RAG Document Q&A With Groq and Llama3")
user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding() 
    st.write("Vector Database is ready")

if user_prompt:
    if st.session_state.vectors is None:
        st.warning("Please click 'Document Embedding' first to build the vector database.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(f"Response time: {time.process_time() - start:.2f} seconds")

        st.write(response['answer'])

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response.get('context', [])):
                st.write(doc.page_content)
                st.write('------------------------')
