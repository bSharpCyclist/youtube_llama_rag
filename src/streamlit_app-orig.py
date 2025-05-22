import streamlit as st
from pathlib import Path
import os
from llama_utils import LlamaIndexRAG
from youtube_utils import save_transcripts_from_playlist, save_transcripts_from_video_list

# Initialize paths
base_transcript_dir = "data/transcripts"
base_storage_dir = "data/storage"

# Ensure directories exist
os.makedirs(base_transcript_dir, exist_ok=True)
os.makedirs(base_storage_dir, exist_ok=True)

# Streamlit UI
st.title("YouTube Transcript RAG Chatbot")

# Sidebar for setup
st.sidebar.header("Setup & Index")

# Load existing indexes
index_folders = ["Select an index..."] + [f.name for f in Path(base_storage_dir).iterdir() if (f / 'vector' / 'docstore.json').exists()]
selected_index = st.sidebar.selectbox("Select an index to load", index_folders)

if st.sidebar.button("Load Index"):
    if selected_index != "Select an index...":
        transcript_dir = os.path.join(base_transcript_dir, selected_index)
        storage_dir = os.path.join(base_storage_dir, selected_index, 'vector')
        llama_rag = LlamaIndexRAG(transcript_dir=transcript_dir, storage_dir=storage_dir)
        st.sidebar.success(f"Index '{selected_index}' loaded successfully.")
    else:
        st.sidebar.warning("Please select a valid index.")

# Chat interface
st.header("Chat")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question about the YouTube videos:")

if st.button("Submit") and user_input:
    if 'llama_rag' in locals():
        response, source_nodes = llama_rag.query(user_input)
        st.session_state.chat_history.append((user_input, response.response))
        st.write(response.response)
        st.markdown("### Sources")
        for source in source_nodes:
            st.markdown(f"- **{source.node.metadata.get('episode_title', 'Unknown')}** (Score: {source.score:.4f})")
    else:
        st.warning("Please load an index first.")

# Display chat history
st.header("Chat History")
for question, answer in st.session_state.chat_history:
    st.markdown(f"**Q:** {question}")
    st.markdown(f"**A:** {answer}")
