import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import chromadb
import sys

def main():
    # Check if a query was provided as a command-line argument
    if len(sys.argv) < 2:
        print("Please provide a query. Example: python query.py 'What does the SOP say about safety procedures?'")
        return

    # Get the query from command-line arguments
    query_text = ' '.join(sys.argv[1:])
    
    # Set up persistent storage and load the index
    PERSIST_DIR = "db"
    
    # Set up Chroma
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection("sop_documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Set up models
    Settings.llm = Ollama(model="mistral")
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Load the index
    index = load_index_from_storage(storage_context)
    
    # Create a query engine
    query_engine = index.as_query_engine()
    
    # Run the query
    print(f"Query: {query_text}")
    print("Searching for answer...")
    response = query_engine.query(query_text)
    
    print("\nAnswer:")
    print(response)

if __name__ == "__main__":
    main()