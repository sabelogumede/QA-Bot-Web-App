import os
from dotenv import load_dotenv

# Load environment variables from .env file (if you use one)
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import SecretStr
# from langchain.schema import Document
from langchain_core.documents import Document
from typing import List
from langchain.embeddings.base import Embeddings  # Add this import if not present
from typing import Any
# import shutil

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import gradio as gr

# --- Google API Key Configuration ---
# It's highly recommended to use environment variables for your API key.
# Example: Set this in a .env file or directly in your shell:
# export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

# Replace with your actual Google API Key or load from environment
GOOGLE_API_KEY = SecretStr(os.getenv("GOOGLE_API_KEY", ""))

# Ensure you have a directory for uploaded PDFs
PDF_UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(PDF_UPLOAD_DIR, exist_ok=True)

# --- 1. Load document using LangChain for different sources ---
def document_loader(file_path: str):
    """
    Implements the document_loader function using PyPDFLoader from the langchain_community library
    to load PDF files.
    """
    print(f"[STEP] Loading document from: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"[STEP] Loaded {len(documents)} pages from PDF.")
    return documents

# --- 2. Splitting long documents using text splitters ---
def text_splitter(documents: List[Document]):
    """
    Completes the text_splitter function using RecursiveCharacterTextSplitter
    to split the loaded PDF content into manageable text chunks.
    """
    print("[STEP] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[STEP] Split into {len(chunks)} chunks.")
    return chunks

# --- 3. Generating embeddings using embedding models (Now Google) ---
def google_embedding():
    """
    Completes the google_embedding() function using the GoogleGenerativeAIEmbeddings class
    from the langchain_google_genai library to generate text embeddings.
    """
    print("[STEP] Initializing GoogleGenerativeAIEmbeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # Or "gemini-embedding-001"
        google_api_key=GOOGLE_API_KEY
    )
    print("[STEP] Embeddings model initialized.")
    return embeddings

# --- 4. Storing embeddings using vector databases ---
# def vector_database(chunks, embeddings):
def vector_database(chunks: List[Document], embeddings: Embeddings):
    """
    Completes the vector_database() function to embed the text chunks using the
    embedding model and store them in a Chroma vector store using Chroma.from_documents().
    """
    print("[STEP] Creating Chroma vector store...")
    # vectordb = Chroma.from_documents(
    #     documents=chunks,
    #     embedding=embeddings
    # )
    vectordb: Chroma = Chroma.from_documents(# type: ignore
    documents=chunks,
    embedding=embeddings
    )
    print("[STEP] Vector database created.")
    return vectordb

# --- 5. Defining retrievers ---
def retriever(file_path: str):
    """
    Completes the retriever(file) function to load, split, embed, and convert documents
    into a retriever using similarity search from a Chroma vector store.
    """
    print(f"[STEP] Setting up retriever for: {file_path}")
    documents = document_loader(file_path)
    chunks = text_splitter(documents)
    embeddings = google_embedding() # Use Google embeddings
    vectordb = vector_database(chunks, embeddings)
    retriever_instance = vectordb.as_retriever(search_kwargs={"k": 3})
    print("[STEP] Retriever defined.")
    return retriever_instance

# --- Helper function to get LLM (Now Google) ---
def get_llm():
    """
    Initializes and returns a ChatGoogleGenerativeAI instance.
    """
    print("[STEP] Initializing ChatGoogleGenerativeAI...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",  # Or "gemini-1.5-pro" or other suitable Gemini models
        google_api_key=GOOGLE_API_KEY.get_secret_value()
    )
    print("[STEP] LLM initialized.")
    return llm

# --- 6. Setting up Gradio as the front-end interface ---
def retriever_qa(file: Any, query: str):
    """
    Defines the retriever_qa(file, query) function using the RetrievalQA chain from langchain
    to perform question-answering over documents using RAG.
    """
    if file is None:
        return "Please upload a PDF document first."
    
    file_path = os.path.join(PDF_UPLOAD_DIR, os.path.basename(file.name))
    
    # Gradio provides a NamedTemporaryFile object, so we need to save it to a permanent location
    # to be read by PyPDFLoader
    with open(file_path, "wb") as f:
        f.write(file.read())
    
    print(f"Processing query '{query}' for file: {file_path}")
    
    llm = get_llm()
    retriever_instance = retriever(file_path) # Call the retriever function

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" combines all retrieved documents into a single prompt
        retriever=retriever_instance,
        return_source_documents=True # Optional: to see which documents were used
    )

    try:
        response = qa_chain.invoke({"query": query})
        answer = response["result"]
        # source_documents = response.get("source_documents", [])
        # sources = "\nSources:\n" + "\n".join([doc.metadata.get("source", "Unknown") for doc in source_documents])
        return answer
    except Exception as e:
        return f"An error occurred: {e}"


# --- Gradio Interface Setup ---
print("Setting up Gradio interface...")
iface = gr.Interface(
    fn=retriever_qa,
    inputs=[
        # gr.File(label="Upload PDF Document", file_types=[".pdf"], type="file"),
        gr.File(label="Upload PDF Document", file_types=[".pdf"], type="filepath"),
        # gr.File(label="Upload PDF Document", file_types=[".pdf"], type="binary"),
        gr.Textbox(lines=2, placeholder="Enter your query here...", label="Your Question")
    ],
    outputs="text",
    title="QA Bot Web App (Powered by Google Gemini)",
    description="Ask questions about your uploaded PDF documents.",
    live=False,
)

if __name__ == "__main__":
    print("Starting Gradio app...")
    iface.launch(share=True)