#!/usr/bin/env python3
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

def main():
    # Set up directories
    DATA_DIR = "data"
    PERSIST_DIR = "db"
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR} directory. Please add your SOP documents there.")
        exit()
    
    # Load documents
    print("Loading documents...")
    documents = SimpleDirectoryReader(DATA_DIR).load_documents()
    
    # Check if any documents were found
    if len(documents) == 0:
        print(f"No documents found in {DATA_DIR}. Please add your SOP documents there.")
        exit()
    print(f"Loaded {len(documents)} documents")
    
    # Set up persistent storage
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)
    
    # Set up Chroma
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection("sop_documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Set up LLM (used for text splitting decisions)
    Settings.llm = Ollama(model="mistral")
    Settings.chunk_size = 512  # Adjust based on your documents
    
    # Create index
    print("Creating index...")
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    print("Ingestion complete! Your documents are now indexed.")

if __name__ == "__main__":
    main()