import os
from dotenv import load_dotenv
import time # For simulating streaming delay

# Load environment variables from .env file
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import gradio as gr
# from langchain.schema import Document
from langchain_core.documents import Document
from typing import List
from pydantic import SecretStr
from langchain.embeddings.base import Embeddings  # Add this import if not present
from typing import Any

# --- Google API Key Configuration ---
GOOGLE_API_KEY = SecretStr(os.getenv("GOOGLE_API_KEY", "")) # Replace if not using .env

# Ensure you have a directory for uploaded PDFs
PDF_UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(PDF_UPLOAD_DIR, exist_ok=True)

# Global variables to store retriever and file path after upload for reuse
# (In a production app, you might use a more robust caching or session management)
current_retriever = None
current_pdf_path = None

# --- 1. Load document using LangChain for different sources ---
def document_loader(file_path: str):
    """
    Loads PDF files using PyPDFLoader.
    """
    print(f"[STEP] Loading document from: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"[STEP] Loaded {len(documents)} pages from PDF.")
    return documents

# --- 2. Splitting long documents using text splitters ---
def text_splitter(documents: List[Document]):
    """
    Splits the loaded PDF content into manageable text chunks.
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

# --- 3. Generating embeddings using embedding models (Google) ---
def google_embedding():
    """
    Initializes GoogleGenerativeAIEmbeddings to generate text embeddings.
    """
    print("[STEP] Initializing GoogleGenerativeAIEmbeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    print("[STEP] Embeddings model initialized.")
    return embeddings

# --- 4. Storing embeddings using vector databases ---
def vector_database(chunks: List[Document], embeddings: Embeddings):
    """
    Embeds text chunks and stores them in a Chroma vector store.
    """
    print("[STEP] Creating Chroma vector store...")
    vectordb = Chroma.from_documents(# type: ignore
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("[STEP] Vector database created.")
    return vectordb

# --- 5. Defining retrievers ---
def create_retriever(file_path: str):
    """
    Loads, splits, embeds, and converts documents into a retriever.
    This function will be called once per PDF upload.
    """
    print(f"[STEP] Setting up retriever for: {file_path}")
    documents = document_loader(file_path)
    chunks = text_splitter(documents)
    embeddings = google_embedding()
    vectordb = vector_database(chunks, embeddings)
    retriever_instance = vectordb.as_retriever(search_kwargs={"k": 3})
    print("[STEP] Retriever defined.")
    return retriever_instance

# --- Helper function to get LLM (Google) ---
def get_llm(system_message: str = ""):
    """
    Initializes and returns a ChatGoogleGenerativeAI instance.
    Includes an optional system message for context.
    """
    print("Initializing ChatGoogleGenerativeAI...")
    # Note: For system messages with ChatGoogleGenerativeAI, you often pass it
    # as part of the messages list in the invoke method, rather than a direct parameter.
    # However, if the LLM itself supports a system prompt parameter, you'd put it here.
    # For simplicity, we'll keep the direct LLM instantiation here.
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", # Ensure this model supports chat and is suitable for your region
        google_api_key=GOOGLE_API_KEY.get_secret_value()
    )
    return llm

# --- Main Chat Function for Gradio ChatInterface ---
def qa_chat_bot(message: str, _history: list[Any], uploaded_file_obj: Any, system_message: str):
    """
    Performs question-answering over documents using RAG with streaming.
    Adapts for gr.ChatInterface.
    """
    global current_retriever, current_pdf_path

    # Handle PDF upload: if a new file is uploaded, process it
    if uploaded_file_obj is not None and uploaded_file_obj.name != current_pdf_path:
        print(f"New PDF uploaded: {uploaded_file_obj.name}")
        file_path = os.path.join(PDF_UPLOAD_DIR, os.path.basename(uploaded_file_obj.name))
        with open(file_path, "wb") as f:
            f.write(uploaded_file_obj.read())
        
        # Update global state with the new file and its retriever
        current_pdf_path = file_path
        current_retriever = create_retriever(file_path)
        yield "Document processed. You can now ask questions."
        return

    # If no file is uploaded yet, or the last processing failed
    if current_retriever is None:
        yield "Please upload a PDF document first and wait for it to process."
        return
    
    # Initialize LLM with an optional system message context (might need to be handled within chain)
    # For Gemini-pro via langchain, a 'system' role is often part of the history or prompt.
    llm = get_llm(system_message) # System message might need more careful integration depending on LLM.

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=current_retriever,
        return_source_documents=False # Set to False for simpler streamed output
    )

    try:
        # LangChain's RetrievalQA.from_chain_type does not inherently support streaming
        # directly in the invoke method. To simulate streaming, we'll get the full answer
        # and then yield it character by character. For true LLM streaming, you'd use
        # a different chain type or directly interact with the LLM's stream method.
        print(f"[STEP] Processing query '{message}'...")
        response_dict = qa_chain.invoke({"query": message})
        full_answer = response_dict.get("result", "No answer found.")

        # Simulate streaming by yielding character by character
        for char in full_answer:
            yield char
            time.sleep(0.01) # Small delay to make streaming visible

    except Exception as e:
        error_message = f"An error occurred: {e}. Please check your API key and PDF."
        print(error_message)
        yield error_message


# --- Gradio ChatInterface Setup ---
print("Setting up Gradio ChatInterface...")

# Custom component for PDF upload, separate from the chat input
pdf_upload_component = gr.File(
    label="Upload PDF Document (Required before chat)",
    file_types=[".pdf"],
    type="filepath"
)

# Additional input for system message
system_message_textbox = gr.Textbox(
    label="System Message (Optional)",
    placeholder="e.g., Act as an expert in history.",
    lines=1,
    value="You are a helpful QA assistant. Answer questions based on the provided document.",
    interactive=True
)

iface = gr.ChatInterface(
    fn=qa_chat_bot,
    # --- FIX HERE: Add type='messages' to gr.Chatbot ---
    chatbot=gr.Chatbot(height=500, type='messages'), # Customize chatbot height
    # --- END FIX ---
    textbox=gr.Textbox(placeholder="Ask me a question about the PDF...", container=False, scale=7),
    additional_inputs=[pdf_upload_component, system_message_textbox],
    examples=[
        ["What is this document about?"],
        ["Summarize the key findings."],
        ["What is the main conclusion?"],
    ],
    cache_examples=False,
    save_history=True,
    # flagging_enabled=True,
    flagging_dir="flagged_data",
    # flagging_label="Flag as Incorrect/Helpful",
    title="QA Bot Web App (Powered by Google Gemini)",
    description="Upload a PDF document, then ask questions about its content. "
                "The bot will stream its answers.",
    theme="soft"
)

if __name__ == "__main__":
    print("Starting Gradio app...")
    iface.launch(share=True)