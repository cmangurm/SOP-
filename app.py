import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os

# Check if database exists
PERSIST_DIR = "db"
if not os.path.exists(PERSIST_DIR):
    st.error("Database not found. Please run ingest.py first.")
    st.stop()

# App title
st.title("SOP Chatbot")
st.write("Ask questions about your Standard Operating Procedures")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to load query engine
@st.cache_resource
def load_query_engine():
    # Set up Chroma
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection("sop_documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load the index
    index = load_index_from_storage(storage_context)
    
    # Create query engine
    return index.as_query_engine(
        llm=Ollama(model="mistral"),
        similarity_top_k=3,
    )

# Get user input
if prompt := st.chat_input("Ask about your SOPs"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from query engine
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            query_engine = load_query_engine()
            response = query_engine.query(prompt)
            
            # Create formatted response with sources
            response_text = response.response
            source_text = "\n\n**Sources:**\n"
            for source_node in response.source_nodes:
                source_text += f"- {source_node.node.metadata.get('file_name', 'Unknown')}\n"
            
            full_response = response_text + source_text
            st.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})