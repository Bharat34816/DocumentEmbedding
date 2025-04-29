import os
import streamlit as st
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# Fix SSL_CERT_FILE issue (delete if set)
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]


st.title("Document Q&A using Llama")


llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")


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

# Function to create vector store
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("data/")  # Add your PDFs here
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")


question = st.text_input("Question from the documents")


if question:
    if "vectors" not in st.session_state:
        st.warning("Please embed documents first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': question})
        end = time.process_time()

        st.subheader("Answer:")
        st.write(response["answer"])
        st.caption(f"Response Time: {round(end - start, 2)} seconds")

        
        with st.expander("Document Similarity Search (Context)"):
            docs = retriever.get_relevant_documents(question)
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.markdown("---")
