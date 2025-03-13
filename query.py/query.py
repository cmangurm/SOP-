from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Set up persistent storage
PERSIST_DIR = "db"

# Set up Chroma
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
chroma_collection = client.get_or_create_collection("sop_documents")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load the index
index = load_index_from_storage(storage_context)

# Create query engine with Ollama
query_engine = index.as_query_engine(
    llm=Ollama(model="mistral"),
    similarity_top_k=3,  # Number of chunks to retrieve
)

# Simple CLI query loop
def main():
    print("SOP Chatbot initialized. Type 'exit' to quit.")
    while True:
        query = input("\nAsk a question about your SOPs: ")
        if query.lower() == 'exit':
            break
        
        response = query_engine.query(query)
        print("\nResponse:", response.response)
        print("\nSource documents:")
        for source_node in response.source_nodes:
            print(f"- {source_node.node.metadata.get('file_name', 'Unknown file')}")

if __name__ == "__main__":
    main()