# streamlit_app.py
import streamlit as st
import os
import logging
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

# Assuming youtube_utils.py and llama_utils.py are in the same directory
# If they are in a sub-directory, adjust the import path
try:
    from youtube_utils import save_transcripts_from_playlist, save_transcripts_from_video_list # extract_video_id not used directly here
    from llama_utils import LlamaIndexRAG
except ImportError as e:
    st.error(f"Failed to import local modules (youtube_utils, llama_utils): {e}. "
             "Ensure they are in the same directory as the Streamlit app or in the Python path.")
    st.stop()


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeRAGStreamlit:
    def __init__(self):
        """Initialize the YouTubeRAGStreamlit application."""
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY', '')
        
        self.base_transcript_dir = "data/transcripts"
        self.base_storage_dir = "data/storage"
        
        os.makedirs(self.base_transcript_dir, exist_ok=True)
        os.makedirs(self.base_storage_dir, exist_ok=True)

        # Initialize Streamlit session state variables
        if "llama_rag_instance" not in st.session_state:
            st.session_state.llama_rag_instance = None
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [] 
        if "current_index_name" not in st.session_state:
            st.session_state.current_index_name = None
        if "youtube_api_key_st" not in st.session_state:
            st.session_state.youtube_api_key_st = self.youtube_api_key
        # Add OPENAI_API_KEY to session state if you want to manage it via UI
        if "openai_api_key_st" not in st.session_state:
            st.session_state.openai_api_key_st = os.getenv('OPENAI_API_KEY', '')


    def get_transcript_folders(self) -> List[str]:
        try:
            return ["Select a folder..."] + [
                f.name for f in Path(self.base_transcript_dir).iterdir() 
                if f.is_dir() and any(f.glob('*.txt'))
            ]
        except Exception as e:
            logger.error(f"Error getting transcript folders: {str(e)}")
            return ["Select a folder..."]
            
    def get_index_folders(self) -> List[str]:
        try:
            folders = ["Select an index..."]
            storage_path = Path(self.base_storage_dir)
            if storage_path.exists():
                for folder in storage_path.iterdir():
                    # LlamaIndexRAG creates a 'vector' subfolder within the named storage_dir
                    index_vector_path = folder / 'vector'
                    if folder.is_dir() and index_vector_path.exists() and (index_vector_path / 'docstore.json').exists():
                        folders.append(folder.name)
            return folders
        except Exception as e:
            logger.error(f"Error getting index folders: {str(e)}")
            return ["Select an index..."]
            
    def download_transcripts_from_playlist(
        self, playlist_url: str, folder_name: str
    ) -> str:
        if not st.session_state.youtube_api_key_st:
            return "⚠️ YouTube API Key is not set. Please set it in the 'Settings' sidebar or as an environment variable."
        if not playlist_url.strip():
            return "⚠️ Please enter a playlist URL or ID"
        if not folder_name.strip():
            return "⚠️ Please enter a folder name"
            
        playlist_match = re.search(r'list=([^&]+)', playlist_url)
        playlist_id = playlist_match.group(1) if playlist_match else playlist_url.strip()
        
        if not playlist_id:
            return "⚠️ Could not extract playlist ID"
            
        output_dir = os.path.join(self.base_transcript_dir, folder_name.strip())
        
        try:
            with st.spinner(f"Downloading transcripts for playlist {playlist_id} to {folder_name}..."):
                success_count = save_transcripts_from_playlist(st.session_state.youtube_api_key_st, playlist_id, output_dir)
            
            if success_count > 0:
                return f"✅ Successfully downloaded {success_count} transcripts to '{folder_name}/'"
            else:
                return "⚠️ No transcripts were downloaded. Check if playlist exists and has videos with transcripts."
        except Exception as e:
            logger.error(f"Error downloading playlist transcripts: {str(e)}", exc_info=True)
            return f"❌ Error downloading transcripts: {str(e)}"
            
    def download_transcripts_from_videos(
        self, video_urls: str, folder_name: str
    ) -> str:
        if not st.session_state.youtube_api_key_st:
            return "⚠️ YouTube API Key is not set. Please set it in the 'Settings' sidebar or as an environment variable."
        if not video_urls.strip():
            return "⚠️ Please enter at least one video URL"
        if not folder_name.strip():
            return "⚠️ Please enter a folder name"
            
        urls = [url.strip() for url in video_urls.split('\n') if url.strip()]
        if not urls:
            return "⚠️ No valid URLs provided"
            
        output_dir = os.path.join(self.base_transcript_dir, folder_name.strip())
        
        try:
            # Your save_transcripts_from_video_list needs the API key.
            # Assuming it gets it from os.environ or you modify it to accept it.
            # If it relies on os.environ['YOUTUBE_API_KEY'], ensure it's set.
            # For now, let's assume it uses the one set by LlamaIndexRAG or globally.
            # A cleaner way would be to pass it: save_transcripts_from_video_list(st.session_state.youtube_api_key_st, urls, output_dir)
            # For now, assuming your function handles API key internally:
            with st.spinner(f"Downloading transcripts for videos to {folder_name}..."):
                success_count = save_transcripts_from_video_list(urls, output_dir)
            
            if success_count > 0:
                return f"✅ Successfully downloaded {success_count} transcripts to '{folder_name}/'"
            else:
                return "⚠️ No transcripts were downloaded. Check if videos exist and have transcripts."
        except Exception as e:
            logger.error(f"Error downloading video transcripts: {str(e)}", exc_info=True)
            return f"❌ Error downloading transcripts: {str(e)}"

    def create_index(self, folder_name: str) -> str:
        """Creates a new index or refreshes an existing one for the selected transcript folder."""
        if not folder_name or folder_name == "Select a folder...":
            return "⚠️ Please select a transcript folder"
        
        if not st.session_state.openai_api_key_st:
             return "⚠️ OpenAI API Key is not set. Please set it in the 'Settings' sidebar or as an environment variable (OPENAI_API_KEY)."

        transcript_dir = os.path.join(self.base_transcript_dir, folder_name)
        # LlamaIndexRAG creates a 'vector' subfolder based on its logic:
        # os.path.join('data/storage', transcript_base, 'vector')
        # So, the storage_dir passed to LlamaIndexRAG should be the parent of 'vector'
        # or LlamaIndexRAG should be modified if we want to control the 'vector' part from here.
        # Based on LlamaIndexRAG: `self.storage_dir = os.path.join('data/storage', transcript_base, 'vector')`
        # So, we pass the base for `transcript_base` which is `folder_name` here.
        # The `storage_dir` argument to LlamaIndexRAG is the *full path* to the vector store.
        # `self.storage_dir = storage_dir` in LlamaIndexRAG.
        # So, `storage_dir_for_rag` should be `data/storage/folder_name/vector`
        storage_dir_for_rag = os.path.join(self.base_storage_dir, folder_name, 'vector')
        
        status_message = ""
        try:
            with st.spinner(f"Initializing RAG for '{folder_name}'. This may load an existing index or prepare for creation/refresh..."):
                # Set OPENAI_API_KEY environment variable for LlamaIndex if provided via UI
                # LlamaIndexRAG uses load_dotenv(), so this should be picked up if set before instantiation.
                if st.session_state.openai_api_key_st and os.getenv('OPENAI_API_KEY') != st.session_state.openai_api_key_st:
                    os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key_st

                st.session_state.llama_rag_instance = LlamaIndexRAG(
                    transcript_dir=transcript_dir,
                    storage_dir=storage_dir_for_rag
                    # LLM/Embedding models are taken from LlamaIndexRAG defaults or env vars
                )

            if not st.session_state.llama_rag_instance or not st.session_state.llama_rag_instance.index:
                # This means initial load/create in __init__ failed (e.g., empty transcript_dir)
                logger.warning(f"Index for '{folder_name}' was not loaded or created during LlamaIndexRAG instantiation. Attempting refresh/creation now.")
                # Proceed to refresh_index, which will attempt creation if index is None

            with st.spinner(f"Refreshing index for '{folder_name}' to ensure it's up-to-date..."):
                st.session_state.llama_rag_instance.refresh_index() 

            if not st.session_state.llama_rag_instance.index:
                 logger.error(f"Index for '{folder_name}' is still not available after refresh attempt.")
                 st.session_state.llama_rag_instance = None 
                 st.session_state.current_index_name = None
                 return f"❌ Failed to create or refresh index for '{folder_name}'. Ensure transcript files exist in '{transcript_dir}' and are valid."

            st.session_state.current_index_name = folder_name # This index is now active
            stats = st.session_state.llama_rag_instance.get_document_stats()

            if stats:
                status_message = f"✅ Index for '{folder_name}' created/refreshed. {stats['document_count']} docs, {stats['node_count']} chunks. Now active."
            else:
                status_message = f"✅ Index for '{folder_name}' created/refreshed. Now active."
            
        except Exception as e:
            logger.error(f"Error creating/refreshing index for '{folder_name}': {str(e)}", exc_info=True)
            st.session_state.llama_rag_instance = None 
            st.session_state.current_index_name = None
            status_message = f"❌ Error creating/refreshing index for '{folder_name}': {str(e)}"
        return status_message
            
    def load_index(self, index_name: str) -> str:
        """Loads an existing index."""
        if not index_name or index_name == "Select an index...":
            return "⚠️ Please select an index"
        
        if not st.session_state.openai_api_key_st:
             return "⚠️ OpenAI API Key is not set. Please set it in the 'Settings' sidebar or as an environment variable (OPENAI_API_KEY)."

        transcript_dir = os.path.join(self.base_transcript_dir, index_name)
        storage_dir_for_rag = os.path.join(self.base_storage_dir, index_name, 'vector')

        try:
            with st.spinner(f"Loading index '{index_name}'..."):
                # Set OPENAI_API_KEY environment variable for LlamaIndex
                if st.session_state.openai_api_key_st and os.getenv('OPENAI_API_KEY') != st.session_state.openai_api_key_st:
                    os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key_st
                
                st.session_state.llama_rag_instance = LlamaIndexRAG(
                    transcript_dir=transcript_dir, 
                    storage_dir=storage_dir_for_rag
                )

                if not st.session_state.llama_rag_instance.index:
                    st.session_state.llama_rag_instance = None 
                    st.session_state.current_index_name = None
                    logger.error(f"Failed to load index '{index_name}'. LlamaIndexRAG initialization did not result in a loaded index object. "
                                 f"The storage at '{storage_dir_for_rag}' might be missing/corrupt, or transcript dir '{transcript_dir}' was empty if it attempted creation.")
                    return (f"❌ Failed to load index '{index_name}'. No valid index found or created. "
                            f"Ensure the index was previously created and its storage is intact. "
                            f"If it tried to rebuild, ensure transcript files exist.")

                stats = st.session_state.llama_rag_instance.get_document_stats()
            
            st.session_state.current_index_name = index_name
            if stats:
                return f"✅ Index '{index_name}' loaded. {stats['document_count']} docs, {stats['node_count']} chunks. Ready to chat!"
            else:
                return f"✅ Index '{index_name}' loaded. Ready to chat!"
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading index '{index_name}': {str(e)}", exc_info=True)
            st.session_state.llama_rag_instance = None
            st.session_state.current_index_name = None
            return f"❌ An unexpected error occurred while loading index '{index_name}': {str(e)}"
            
    def get_sources_text(self, source_nodes) -> str:
        if not source_nodes:
            return ""
        sources_text = "**Sources:**\n\n"
        for i, source in enumerate(source_nodes):
            # LlamaIndexRAG sets 'episode_title' in metadata
            title = source.node.metadata.get('episode_title', source.node.metadata.get('file_name', f"Source {i+1}"))
            score = source.score if hasattr(source, 'score') else "N/A" # score might not always be present depending on node type
            text = source.node.get_content()
            
            sources_text += f"**Source {i+1}:** {title}  \n"
            if score != "N/A":
                sources_text += f"*Relevance Score: {score:.4f}*\n\n"
            else:
                sources_text += "\n"
            sources_text += f"```text\n{text[:300]}...\n```\n\n" # Using text for code block for better wrapping
        return sources_text
        
    def process_chat_message(self, message: str, top_k: int = 3, show_sources: bool = True) -> Tuple[str, str]:
        """Processes a chat message and returns (response_text, sources_markdown_text)."""
        if not message.strip():
            return "", ""
            
        if not st.session_state.llama_rag_instance or not st.session_state.llama_rag_instance.index:
            return "Please load an index first. Go to the 'Setup & Index' tab.", ""
        
        if not st.session_state.openai_api_key_st: # Check again before query
             return "OpenAI API Key is not set. Cannot query. Please set it in 'Settings' sidebar.", ""
            
        try:
            with st.spinner("Thinking..."):
                # LlamaIndexRAG.query takes similarity_top_k
                response_obj, source_nodes = st.session_state.llama_rag_instance.query(
                    message, 
                    similarity_top_k=top_k
                )
            
            sources_text_md = self.get_sources_text(source_nodes) if show_sources else ""
            # response_obj is LlamaIndex Response object, .response gives the string
            resp_text = response_obj.response if hasattr(response_obj, 'response') and response_obj.response is not None else "No response generated."
            
            return resp_text, sources_text_md
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            error_msg = f"Error processing your query: {str(e)}"
            return error_msg, ""

def run_streamlit_app():
    st.set_page_config(layout="wide", page_title="YouTube RAG Chatbot")
    st.title("📚 YouTube Transcript RAG Chatbot")

    rag_app = YouTubeRAGStreamlit() # Initializes session_state variables

    with st.sidebar:
        st.header("Chat Settings")
        top_k_slider = st.slider("Number of sources (top_k)", 1, 10, 3, key="top_k_slider_main")
        show_sources_checkbox = st.checkbox("Show sources in chat", True, key="show_sources_main")
        
        st.header("API Keys")
        st.session_state.youtube_api_key_st = st.text_input(
            "YouTube API Key", 
            value=st.session_state.get("youtube_api_key_st", ""), 
            type="password",
            help="Set your YouTube Data API v3 Key. Also settable via YOUTUBE_API_KEY env var."
        )
        if not st.session_state.youtube_api_key_st:
            st.warning("YouTube API Key not set. Transcript download will fail.")

        st.session_state.openai_api_key_st = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get("openai_api_key_st", ""),
            type="password",
            help="Set your OpenAI API Key. Also settable via OPENAI_API_KEY env var. Required for indexing and querying."
        )
        if not st.session_state.openai_api_key_st:
            st.warning("OpenAI API Key not set. Indexing and querying will fail.")


        st.header("Loaded Index")
        if st.session_state.current_index_name:
            st.success(f"Active Index: **{st.session_state.current_index_name}**")
            if st.session_state.llama_rag_instance and st.session_state.llama_rag_instance.index:
                stats = st.session_state.llama_rag_instance.get_document_stats()
                if stats:
                    st.caption(f"({stats['document_count']} docs, {stats['node_count']} chunks)")
        else:
            st.info("No index loaded. Go to 'Setup & Index' tab.")

    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "⚙️ Setup & Index", "📥 Download Transcripts", "❓ Help"])

    with tab1: # Chat
        st.header("Chat with Your YouTube Index")

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                    with st.expander("View Sources"):
                        st.markdown(msg["sources"])
        
        if prompt := st.chat_input("Ask a question about the indexed videos...", key="chat_input_main"):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Use values from sidebar directly
                response_text, sources_md = rag_app.process_chat_message(
                    prompt, 
                    st.session_state.top_k_slider_main, 
                    st.session_state.show_sources_main
                )
                st.markdown(response_text)
                if sources_md:
                    with st.expander("View Sources"):
                        st.markdown(sources_md)
            
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": response_text,
                "sources": sources_md
            })
            st.rerun() 

        if st.session_state.chat_messages:
            if st.button("Clear Chat History", key="clear_chat_btn"):
                st.session_state.chat_messages = []
                st.rerun()

    with tab2: # Setup & Index
        st.header("Manage Your RAG Index")
        
        st.subheader("Load Existing Index")
        index_folders = rag_app.get_index_folders()
        
        # Determine default index for selectbox
        current_selection_load = st.session_state.current_index_name
        if current_selection_load and current_selection_load in index_folders:
            default_load_idx = index_folders.index(current_selection_load)
        elif index_folders and index_folders[0] == "Select an index...":
             default_load_idx = 0
        else: # No specific default, or list is empty/doesn't have placeholder
             default_load_idx = 0


        selected_index_to_load = st.selectbox(
            "Select an index to load", 
            options=index_folders,
            key="load_index_select",
            index=default_load_idx
        )
        
        if st.button("Load Selected Index", key="load_idx_btn"):
            if selected_index_to_load != "Select an index...":
                status = rag_app.load_index(selected_index_to_load)
                if status.startswith("✅"): st.success(status)
                elif status.startswith("⚠️"): st.warning(status)
                else: st.error(status)
                st.rerun() 
            else:
                st.warning("Please select an index from the dropdown.")

        st.divider()
        st.subheader("Create or Refresh Index from Transcripts")
        transcript_folders = rag_app.get_transcript_folders()
        selected_transcript_folder = st.selectbox(
            "Select transcript folder to index/refresh", 
            options=transcript_folders,
            key="create_index_select"
        )

        if st.button("Create / Refresh Index", key="create_idx_btn"):
            if selected_transcript_folder != "Select a folder...":
                status = rag_app.create_index(selected_transcript_folder)
                if status.startswith("✅"): st.success(status)
                elif status.startswith("⚠️"): st.warning(status)
                else: st.error(status)
                st.rerun()
            else:
                st.warning("Please select a transcript folder.")
        
        if st.button("Refresh Folder Lists", key="refresh_folders_btn_setup"):
             st.rerun()

    with tab3: # Download Transcripts
        st.header("Download Transcripts")
        
        download_tab1, download_tab2 = st.tabs(["From Playlist", "From Videos"])

        with download_tab1:
            st.subheader("Download from YouTube Playlist")
            playlist_url = st.text_input(
                "YouTube Playlist URL or ID", 
                placeholder="https://www.youtube.com/playlist?list=PLAYLIST_ID or just PLAYLIST_ID",
                key="playlist_url_input"
            )
            playlist_folder_name = st.text_input(
                "Destination Folder Name (for transcripts)", 
                placeholder="e.g., ancient-aliens-playlist",
                key="playlist_folder_input"
            )
            if st.button("Download Playlist Transcripts", key="download_playlist_btn"):
                status = rag_app.download_transcripts_from_playlist(playlist_url, playlist_folder_name)
                if status.startswith("✅"): st.success(status)
                elif status.startswith("⚠️"): st.warning(status)
                else: st.error(status)
                st.rerun() 

        with download_tab2:
            st.subheader("Download from Individual YouTube Videos")
            video_urls = st.text_area(
                "YouTube Video URLs (one per line)",
                placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=...",
                key="video_urls_input",
                height=150
            )
            videos_folder_name = st.text_input(
                "Destination Folder Name (for transcripts)",
                placeholder="e.g., selected-talks",
                key="videos_folder_input"
            )
            if st.button("Download Video Transcripts", key="download_videos_btn"):
                status = rag_app.download_transcripts_from_videos(video_urls, videos_folder_name)
                if status.startswith("✅"): st.success(status)
                elif status.startswith("⚠️"): st.warning(status)
                else: st.error(status)
                st.rerun() 
        
        if st.button("Refresh Transcript Folder List", key="refresh_folders_btn_download"):
             st.rerun()

    with tab4: # Help
        st.header("How to Use This Application")
        st.markdown("""
        ## ⚙️ Initial Setup (API Keys)
        1.  Go to the **sidebar** (usually on the left).
        2.  Under "API Keys":
            *   Enter your **YouTube Data API v3 Key**. Required for downloading transcripts.
            *   Enter your **OpenAI API Key**. Required for creating/refreshing indexes and for querying the AI.
        3.  These keys can also be set as environment variables (`YOUTUBE_API_KEY`, `OPENAI_API_KEY`) before running the app. Keys entered in the UI will take precedence for the session if the environment variable is also set.

        ## 📥 Downloading Transcripts
        1.  Navigate to the "**Download Transcripts**" tab.
        2.  Choose either "**From Playlist**" or "**From Videos**".
        3.  Provide the YouTube URL(s)/ID and a **Destination Folder Name**. This name creates a subfolder under `data/transcripts/`.
        4.  Click "**Download ... Transcripts**".

        ## ⚙️ Creating or Refreshing an Index
        1.  Go to the "**Setup & Index**" tab.
        2.  Under "**Create or Refresh Index from Transcripts**", select a folder from the dropdown. These are folders in `data/transcripts/` containing `.txt` files.
        3.  Click "**Create / Refresh Index**".
            *   If an index for this folder doesn't exist, it will be created.
            *   If an index already exists, it will be refreshed with the current transcripts in the folder.
            *   The created/refreshed index automatically becomes the active one for chatting.
        4.  The storage for indexes is in `data/storage/your-folder-name/vector/`.

        ## ⚙️ Loading an Existing Index
        1.  Go to the "**Setup & Index**" tab.
        2.  Under "**Load Existing Index**", select an index from the dropdown. These are based on existing storage folders.
        3.  Click "**Load Selected Index**". This makes the selected index active for chatting.

        ## 💬 Chatting with Your Videos
        1.  Ensure an index is loaded (check sidebar or "Setup & Index" tab).
        2.  Go to the "**Chat**" tab.
        3.  Type your question and press Enter.
        4.  Adjust chat settings (number of sources, show/hide sources) in the sidebar.

        ## ✨ Tips
        -   If dropdowns don't update after a download or index operation, use the "**Refresh Folder Lists**" or "**Refresh Transcript Folder List**" buttons.
        -   The application state (API keys, loaded index, chat history) is maintained per user session.
        """)

if __name__ == "__main__":
    run_streamlit_app()