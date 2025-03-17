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
    
    if not os.path.exists(PERSIST_DIR):
        print(f"Error: Database directory '{PERSIST_DIR}' not found. Run ingest.py first.")
        return
    
    print(f"Loading index from {PERSIST_DIR}...")
    
    # Set up models before loading from storage
    Settings.llm = Ollama(model="mistral")
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    
    # Set up Chroma
    try:
        chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
        chroma_collection = chroma_client.get_collection("sop_documents")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index from vector store
        from llama_index.core import VectorStoreIndex
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # Create a query engine
        query_engine = index.as_query_engine()
        
        # Run the query
        print(f"Query: {query_text}")
        print("Searching for answer...")
        response = query_engine.query(query_text)
        
        print("\nAnswer:")
        print(response)
    
    except Exception as e:
        print(f"Error: {e}")
        print("Check if the database was properly created with ingest.py.")

if __name__ == "__main__":
    main()