# LumberJack
Logging tool for warehouse management

## Motivation
Warehouse management deals with dense log files to solve shipping fullfillment errors. These log files can be difficult to read and parsing through this information is a struggle, even with screens built to view log file data in real time. LumberJack is an LLM that uses Retrieval Augmented Generation (RAG) for warehouse management to chat with their log files.

## Framework
### Overview
Ollama (LLM Server) -> Langchain (RAG) -> Chroma (vector storage database) -> Streamlit (Chat UI)
### RAG Pipeline
Two Methods:
1. Ingest
    - Accepts a file path
    - Loads file path into vector storage in two steps:
        a) splits document into smaller chunks to accommodate token limit of LLM
        b) vectorizes these chunks using Qdrant FastEmbeddings and store into Chroma.
2. Ask
    - Handles user queries
    - Users can pose a question, then RetrievalQAChain retrieves the relevant contexts (document chunks) using vector similarity search techniques.
    - With user's question and the tretrieved contexts, we can compose a prompt and request a prediction from the LLM server.