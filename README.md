# QA Bot Web App (Powered by Google Gemini)

## Overview
QA Bot Web App is an intelligent, production-ready question-answering chatbot that allows users to upload PDF documents and ask questions about their content. It leverages Google Gemini (via LangChain), Chroma vector database, and Gradio for a seamless, interactive web experience. The app is designed for real-world document analysis, knowledge extraction, and enterprise search use cases.

---

## Key Features
- **PDF Upload & Parsing:** Users can upload PDF files for instant analysis.
- **Chunking & Embedding:** Documents are split into manageable chunks and embedded using Google Gemini embeddings.
- **Vector Search:** Chroma vector store enables fast, semantic retrieval of relevant document sections.
- **Conversational QA:** Users interact with a chatbot UI, powered by Gradio, to ask questions and receive context-aware answers.
- **Streaming Responses:** Answers are streamed for a responsive user experience.
- **Persistent Storage:** Uploaded files and vector stores are persisted for reuse and scalability.
- **Feedback & Flagging:** User feedback is stored for continuous improvement.

---

## Ideal Real-World Use Cases
- **Legal Document Analysis:** Instantly query contracts, case law, or compliance documents.
- **Academic Research:** Summarize, search, and extract insights from scientific papers.
- **Enterprise Knowledge Base:** Internal document search for HR, onboarding, or technical manuals.
- **Healthcare:** Extract and answer questions from medical guidelines or patient records.
- **Customer Support:** Automated FAQ and troubleshooting from product manuals.

---

## Project Structure
```
QA-Bot-Web-App/
├── qa_bot_app.py         # Main application code (Gradio, LangChain, Gemini, Chroma)
├── requirements.txt      # Python dependencies
├── uploaded_pdfs/        # Directory for uploaded PDF files
├── chroma_db/            # Persistent Chroma vector database
├── flagged_data/         # User feedback and flagged responses
└── README.md             # Project documentation
```

---

## Technical Architecture & Flow

### 1. **PDF Upload**
- User uploads a PDF via the Gradio web interface.
- File is saved to `uploaded_pdfs/` for persistent access.

### 2. **Document Loading**
- `document_loader(file_path)` uses `PyPDFLoader` to extract text from the PDF.

### 3. **Text Splitting**
- `text_splitter(documents)` splits the document into overlapping chunks for better context retention.

### 4. **Embedding Generation**
- `google_embedding()` initializes Google Gemini embeddings.
- Chunks are embedded for semantic search.

### 5. **Vector Database Storage**
- `vector_database(chunks, embeddings)` stores embeddings in a persistent Chroma vector store (`chroma_db/`).

### 6. **Retriever Setup**
- `create_retriever(file_path)` builds a retriever for similarity search over the vector store.

### 7. **Conversational QA**
- `qa_chat_bot(message, history, uploaded_file_obj, system_message)` handles chat logic:
  - On new PDF upload: processes and indexes the document.
  - On question: retrieves relevant chunks and queries Gemini LLM for an answer.
  - Streams the answer back to the user.

### 8. **Feedback & Flagging**
- User can flag responses; feedback is stored in `flagged_data/` for review.

---

## Functionality Testing
- **Unit Tests:**
  - Test each function (document loading, splitting, embedding, retrieval) with sample PDFs.
  - Mock LLM and embedding calls for isolated testing.
- **Integration Tests:**
  - Upload a PDF and verify end-to-end QA flow.
  - Test error handling (e.g., missing API key, corrupt PDF).
- **Manual Testing:**
  - Use the Gradio UI to upload, ask questions, and flag responses.

---

## Deployment Guide

### 1. **Local Development**
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Run the app:
  ```bash
  python qa_bot_app.py
  ```
- Access at `http://127.0.0.1:7860/` (default Gradio port).

### 2. **Dockerization**
- Create a `Dockerfile`:
  ```dockerfile
  FROM python:3.10-slim
  WORKDIR /app
  COPY . .
  RUN pip install --no-cache-dir -r requirements.txt
  EXPOSE 7860
  CMD ["python", "qa_bot_app.py"]
  ```
- Build and run:
  ```bash
  docker build -t qa-bot-web-app .
  docker run -p 7860:7860 qa-bot-web-app
  ```

### 3. **CI/CD Pipeline**
- Use GitHub Actions, GitLab CI, or similar to automate:
  - Linting and testing on push/PR.
  - Build and push Docker images to a registry.
  - Deploy to your cloud provider (e.g., AWS ECS, GCP Cloud Run, Azure App Service).

### 4. **Production Deployment & Scaling**
- **Reverse Proxy:** Use Nginx or Traefik to route traffic and enable HTTPS.
- **Scaling:**
  - Run multiple containers behind a load balancer for high availability.
  - Use persistent storage for `uploaded_pdfs/` and `chroma_db/` (e.g., AWS EFS, GCP Filestore).
  - Monitor resource usage and autoscale as needed.
- **Environment Variables:** Store secrets (API keys) securely using environment variables or a secrets manager.
- **Observability:** Integrate logging and monitoring (e.g., Prometheus, Grafana, Sentry).

---

## References
- [Gradio: Creating a Chatbot Fast](https://www.gradio.app/guides/creating-a-chatbot-fast)
- [Gemini + LangChain + Chroma QA Example (Colab)](https://colab.research.google.com/github/google/generative-ai-docs/blob/main/examples/gemini/python/langchain/Gemini_LangChain_QA_Chroma_WebLoad.ipynb)

---

## License
This project is licensed under the MIT License.
