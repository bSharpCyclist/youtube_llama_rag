# streamlit_app.py
import streamlit as st
import os
import logging
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

try:
    from youtube_utils import save_transcripts_from_playlist, save_transcripts_from_video_list
    from llama_utils_llms import LlamaIndexRAG
except ImportError as e:
    st.error(f"Failed to import local modules (youtube_utils, llama_utils): {e}. "
             "Ensure they are in the same directory as the Streamlit app or in the Python path.")
    st.stop()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeRAGStreamlit:
    def __init__(self):
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY', '')
        self.base_transcript_dir = "data/transcripts"
        self.base_storage_dir = "data/storage"
        
        os.makedirs(self.base_transcript_dir, exist_ok=True)
        os.makedirs(self.base_storage_dir, exist_ok=True)

        if "llama_rag_instance" not in st.session_state:
            st.session_state.llama_rag_instance = None
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [] 
        if "current_index_name" not in st.session_state:
            st.session_state.current_index_name = None
        if "youtube_api_key_st" not in st.session_state:
            st.session_state.youtube_api_key_st = self.youtube_api_key
        if "openai_api_key_st" not in st.session_state:
            st.session_state.openai_api_key_st = os.getenv('OPENAI_API_KEY', '')
        # ### MODIFIED SECTION START ###
        if "google_api_key_st" not in st.session_state: # NEW
            st.session_state.google_api_key_st = os.getenv('GOOGLE_API_KEY', '') # NEW
        if "llm_provider_st" not in st.session_state: # NEW
            st.session_state.llm_provider_st = "openai" # NEW: Default to OpenAI
        # ### MODIFIED SECTION END ###

    def get_transcript_folders(self) -> List[str]:
        # ... (no changes in this method)
        try:
            return ["Select a folder..."] + [
                f.name for f in Path(self.base_transcript_dir).iterdir() 
                if f.is_dir() and any(f.glob('*.txt'))
            ]
        except Exception as e:
            logger.error(f"Error getting transcript folders: {str(e)}")
            return ["Select a folder..."]
            
    def get_index_folders(self) -> List[str]:
        # ... (no changes in this method)
        try:
            folders = ["Select an index..."]
            storage_path = Path(self.base_storage_dir)
            if storage_path.exists():
                for folder in storage_path.iterdir():
                    index_vector_path = folder / 'vector'
                    if folder.is_dir() and index_vector_path.exists() and (index_vector_path / 'docstore.json').exists():
                        folders.append(folder.name)
            return folders
        except Exception as e:
            logger.error(f"Error getting index folders: {str(e)}")
            return ["Select an index..."]
            
    def download_transcripts_from_playlist(self, playlist_url: str, folder_name: str) -> str:
        # ... (no changes in this method, but ensure API key check is robust)
        if not st.session_state.youtube_api_key_st:
            return "⚠️ YouTube API Key is not set. Please set it in the 'Settings' sidebar or as an environment variable."
        # ... (rest of the method)
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

    def download_transcripts_from_videos(self, video_urls: str, folder_name: str) -> str:
        # ... (no changes in this method, but ensure API key check is robust)
        if not st.session_state.youtube_api_key_st:
            return "⚠️ YouTube API Key is not set. Please set it in the 'Settings' sidebar or as an environment variable."
        # ... (rest of the method)
        if not video_urls.strip():
            return "⚠️ Please enter at least one video URL"
        if not folder_name.strip():
            return "⚠️ Please enter a folder name"
            
        urls = [url.strip() for url in video_urls.split('\n') if url.strip()]
        if not urls:
            return "⚠️ No valid URLs provided"
            
        output_dir = os.path.join(self.base_transcript_dir, folder_name.strip())
        
        try:
            with st.spinner(f"Downloading transcripts for videos to {folder_name}..."):
                success_count = save_transcripts_from_video_list(urls, output_dir)
            
            if success_count > 0:
                return f"✅ Successfully downloaded {success_count} transcripts to '{folder_name}/'"
            else:
                return "⚠️ No transcripts were downloaded. Check if videos exist and have transcripts."
        except Exception as e:
            logger.error(f"Error downloading video transcripts: {str(e)}", exc_info=True)
            return f"❌ Error downloading transcripts: {str(e)}"

    # ### MODIFIED SECTION START ###
    def _prepare_api_keys_for_rag(self) -> bool: # NEW helper method
        """Sets API keys in environment if provided via UI and selected provider needs them."""
        provider = st.session_state.llm_provider_st
        if provider == "openai":
            if not st.session_state.openai_api_key_st:
                st.error("OpenAI API Key is not set. Please set it in the 'Settings' sidebar.")
                return False
            if os.getenv('OPENAI_API_KEY') != st.session_state.openai_api_key_st:
                os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key_st
        elif provider == "gemini":
            if not st.session_state.google_api_key_st:
                st.error("Google API Key is not set for Gemini. Please set it in the 'Settings' sidebar.")
                return False
            if os.getenv('GOOGLE_API_KEY') != st.session_state.google_api_key_st:
                os.environ['GOOGLE_API_KEY'] = st.session_state.google_api_key_st
        return True

    def create_index(self, folder_name: str) -> str:
        if not folder_name or folder_name == "Select a folder...":
            return "⚠️ Please select a transcript folder"
        
        if not self._prepare_api_keys_for_rag(): # MODIFIED: Use helper
            return "⚠️ API Key for the selected LLM provider is missing."

        transcript_dir = os.path.join(self.base_transcript_dir, folder_name)
        storage_dir_for_rag = os.path.join(self.base_storage_dir, folder_name, 'vector')
        
        status_message = ""
        try:
            with st.spinner(f"Initializing RAG for '{folder_name}' using {st.session_state.llm_provider_st}..."):
                st.session_state.llama_rag_instance = LlamaIndexRAG(
                    transcript_dir=transcript_dir,
                    storage_dir=storage_dir_for_rag,
                    llm_provider=st.session_state.llm_provider_st # NEW: Pass provider
                    # Model names will use defaults in LlamaIndexRAG or env vars
                )

            if not st.session_state.llama_rag_instance or not st.session_state.llama_rag_instance.index:
                logger.warning(f"Index for '{folder_name}' not loaded/created on init. Attempting refresh/creation.")

            with st.spinner(f"Refreshing index for '{folder_name}'..."):
                st.session_state.llama_rag_instance.refresh_index() 

            if not st.session_state.llama_rag_instance.index:
                 logger.error(f"Index for '{folder_name}' still not available after refresh.")
                 st.session_state.llama_rag_instance = None 
                 st.session_state.current_index_name = None
                 return f"❌ Failed to create or refresh index for '{folder_name}'. Ensure transcript files exist and API keys are correct."

            st.session_state.current_index_name = folder_name
            stats = st.session_state.llama_rag_instance.get_document_stats()
            status_message = f"✅ Index for '{folder_name}' ({st.session_state.llm_provider_st}) "
            if stats:
                status_message += f"created/refreshed. {stats['document_count']} docs, {stats['node_count']} chunks. Now active."
            else:
                status_message += "created/refreshed. Now active."
            
        except Exception as e:
            logger.error(f"Error creating/refreshing index for '{folder_name}': {str(e)}", exc_info=True)
            st.session_state.llama_rag_instance = None 
            st.session_state.current_index_name = None
            status_message = f"❌ Error creating/refreshing index for '{folder_name}': {str(e)}"
        return status_message
            
    def load_index(self, index_name: str) -> str:
        if not index_name or index_name == "Select an index...":
            return "⚠️ Please select an index"
        
        if not self._prepare_api_keys_for_rag(): # MODIFIED: Use helper
            return "⚠️ API Key for the selected LLM provider is missing."

        transcript_dir = os.path.join(self.base_transcript_dir, index_name)
        storage_dir_for_rag = os.path.join(self.base_storage_dir, index_name, 'vector')

        try:
            with st.spinner(f"Loading index '{index_name}' using {st.session_state.llm_provider_st}..."):
                st.session_state.llama_rag_instance = LlamaIndexRAG(
                    transcript_dir=transcript_dir, 
                    storage_dir=storage_dir_for_rag,
                    llm_provider=st.session_state.llm_provider_st # NEW: Pass provider
                )

                if not st.session_state.llama_rag_instance.index:
                    st.session_state.llama_rag_instance = None 
                    st.session_state.current_index_name = None
                    logger.error(f"Failed to load index '{index_name}'.")
                    return (f"❌ Failed to load index '{index_name}'. No valid index found or created. "
                            f"Ensure storage is intact and API keys for {st.session_state.llm_provider_st} are correct.")

                stats = st.session_state.llama_rag_instance.get_document_stats()
            
            st.session_state.current_index_name = index_name
            status_message = f"✅ Index '{index_name}' ({st.session_state.llm_provider_st}) loaded. "
            if stats:
                status_message += f"{stats['document_count']} docs, {stats['node_count']} chunks. Ready to chat!"
            else:
                status_message += "Ready to chat!"
            return status_message
        except Exception as e:
            logger.error(f"Error loading index '{index_name}': {str(e)}", exc_info=True)
            st.session_state.llama_rag_instance = None
            st.session_state.current_index_name = None
            return f"❌ Error loading index '{index_name}': {str(e)}"
    # ### MODIFIED SECTION END ###
            
    def get_sources_text(self, source_nodes) -> str:
        # ... (no changes in this method)
        if not source_nodes:
            return ""
        sources_text = "**Sources:**\n\n"
        for i, source in enumerate(source_nodes):
            title = source.node.metadata.get('episode_title', source.node.metadata.get('file_name', f"Source {i+1}"))
            score = source.score if hasattr(source, 'score') else "N/A" 
            text = source.node.get_content()
            
            sources_text += f"**Source {i+1}:** {title}  \n"
            if score != "N/A":
                sources_text += f"*Relevance Score: {score:.4f}*\n\n"
            else:
                sources_text += "\n"
            sources_text += f"```text\n{text[:300]}...\n```\n\n" 
        return sources_text
        
    def process_chat_message(self, message: str, top_k: int = 3, show_sources: bool = True) -> Tuple[str, str]:
        if not message.strip():
            return "", ""
            
        if not st.session_state.llama_rag_instance or not st.session_state.llama_rag_instance.index:
            return "Please load an index first. Go to the 'Setup & Index' tab.", ""
        
        # ### MODIFIED SECTION START ###
        # API key check before query, specific to provider
        provider = st.session_state.llm_provider_st
        api_key_missing_msg = ""
        if provider == "openai" and not st.session_state.openai_api_key_st:
            api_key_missing_msg = "OpenAI API Key is not set. Cannot query."
        elif provider == "gemini" and not st.session_state.google_api_key_st:
            api_key_missing_msg = "Google API Key is not set for Gemini. Cannot query."
        
        if api_key_missing_msg:
            return f"{api_key_missing_msg} Please set it in 'Settings' sidebar.", ""
        # ### MODIFIED SECTION END ###
            
        try:
            with st.spinner(f"Thinking with {provider}..."):
                response_obj, source_nodes = st.session_state.llama_rag_instance.query(
                    message, 
                    similarity_top_k=top_k
                )
            
            sources_text_md = self.get_sources_text(source_nodes) if show_sources else ""
            resp_text = response_obj.response if hasattr(response_obj, 'response') and response_obj.response is not None else "No response generated."
            
            return resp_text, sources_text_md
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            error_msg = f"Error processing your query: {str(e)}"
            return error_msg, ""

def run_streamlit_app():
    st.set_page_config(layout="wide", page_title="YouTube RAG Chatbot")
    st.title("📚 YouTube Transcript RAG Chatbot")

    rag_app = YouTubeRAGStreamlit()

    with st.sidebar:
        st.header("Chat Settings")
        # ### MODIFIED SECTION START ###
        st.session_state.llm_provider_st = st.selectbox( # NEW: LLM Provider selection
            "Choose LLM Provider",
            options=["openai", "gemini"],
            index=["openai", "gemini"].index(st.session_state.get("llm_provider_st", "openai")), # Maintain selection
            key="llm_provider_selector"
        )
        # ### MODIFIED SECTION END ###

        top_k_slider = st.slider("Number of sources (top_k)", 1, 10, 3, key="top_k_slider_main")
        show_sources_checkbox = st.checkbox("Show sources in chat", True, key="show_sources_main")
        
        st.header("API Keys")
        st.session_state.youtube_api_key_st = st.text_input(
            "YouTube API Key", 
            value=st.session_state.get("youtube_api_key_st", ""), 
            type="password",
            help="Set your YouTube Data API v3 Key."
        )
        if not st.session_state.youtube_api_key_st:
            st.warning("YouTube API Key not set. Transcript download will fail.")

        # ### MODIFIED SECTION START ###
        # Conditional API Key inputs based on provider selection
        if st.session_state.llm_provider_st == "openai": # MODIFIED
            st.session_state.openai_api_key_st = st.text_input(
                "OpenAI API Key",
                value=st.session_state.get("openai_api_key_st", ""),
                type="password",
                help="Required if OpenAI is selected as LLM Provider."
            )
            if not st.session_state.openai_api_key_st:
                st.warning("OpenAI API Key not set.")
        
        elif st.session_state.llm_provider_st == "gemini": # NEW
            st.session_state.google_api_key_st = st.text_input( # NEW
                "Google API Key (for Gemini)",
                value=st.session_state.get("google_api_key_st", ""),
                type="password",
                help="Required if Gemini is selected as LLM Provider."
            )
            if not st.session_state.google_api_key_st: # NEW
                st.warning("Google API Key for Gemini not set.")
        # ### MODIFIED SECTION END ###

        st.header("Loaded Index")
        if st.session_state.current_index_name:
            # ### MODIFIED SECTION START ###
            provider_in_use = st.session_state.llama_rag_instance.llm_provider if st.session_state.llama_rag_instance else "N/A"
            st.success(f"Active Index: **{st.session_state.current_index_name}** (Provider: {provider_in_use.capitalize()})")
            # ### MODIFIED SECTION END ###
            if st.session_state.llama_rag_instance and st.session_state.llama_rag_instance.index:
                stats = st.session_state.llama_rag_instance.get_document_stats()
                if stats:
                    st.caption(f"({stats['document_count']} docs, {stats['node_count']} chunks)")
        else:
            st.info("No index loaded. Go to 'Setup & Index' tab.")

    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "⚙️ Setup & Index", "📥 Download Transcripts", "❓ Help"])

    with tab1: # Chat
        # ... (Chat message display loop - no changes)
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
        st.header(f"Manage Your RAG Index (Using: {st.session_state.llm_provider_st.capitalize()})") # MODIFIED: Show current provider
        
        st.subheader("Load Existing Index")
        index_folders = rag_app.get_index_folders()
        current_selection_load = st.session_state.current_index_name
        default_load_idx = 0
        if current_selection_load and current_selection_load in index_folders:
            default_load_idx = index_folders.index(current_selection_load)
        elif index_folders and index_folders[0] == "Select an index...":
             default_load_idx = 0
        
        selected_index_to_load = st.selectbox(
            "Select an index to load", 
            options=index_folders,
            key="load_index_select",
            index=default_load_idx
        )
        
        if st.button("Load Selected Index", key="load_idx_btn"):
            if selected_index_to_load != "Select an index...":
                # ### MODIFIED SECTION START ###
                # When loading, the index was built with a specific provider's embeddings.
                # Ideally, the UI should only show indexes compatible with the selected provider,
                # or the LlamaIndexRAG should handle loading an index and checking compatibility.
                # For now, it will attempt to load using the currently selected provider.
                # This might fail if embeddings are incompatible.
                st.info(f"Attempting to load index '{selected_index_to_load}' using {st.session_state.llm_provider_st.capitalize()} embeddings/LLM.")
                # ### MODIFIED SECTION END ###
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
                # ### MODIFIED SECTION START ###
                st.info(f"Creating/refreshing index for '{selected_transcript_folder}' using {st.session_state.llm_provider_st.capitalize()} embeddings/LLM.")
                # ### MODIFIED SECTION END ###
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
        # ... (no changes in this tab)
        st.header("Download Transcripts")
        download_tab1, download_tab2 = st.tabs(["From Playlist", "From Videos"])
        with download_tab1:
            st.subheader("Download from YouTube Playlist")
            playlist_url = st.text_input("YouTube Playlist URL or ID", placeholder="...", key="playlist_url_input")
            playlist_folder_name = st.text_input("Destination Folder Name (for transcripts)", placeholder="...", key="playlist_folder_input")
            if st.button("Download Playlist Transcripts", key="download_playlist_btn"):
                status = rag_app.download_transcripts_from_playlist(playlist_url, playlist_folder_name)
                if status.startswith("✅"):
                    st.success(status)
                elif status.startswith("⚠️"):
                    st.warning(status)
                else:
                    st.error(status)
                st.rerun() 
        with download_tab2:
            st.subheader("Download from Individual YouTube Videos")
            video_urls = st.text_area("YouTube Video URLs (one per line)", placeholder="...", key="video_urls_input", height=150)
            videos_folder_name = st.text_input("Destination Folder Name (for transcripts)", placeholder="...", key="videos_folder_input")
            if st.button("Download Video Transcripts", key="download_videos_btn"):
                status = rag_app.download_transcripts_from_videos(video_urls, videos_folder_name)
                if status.startswith("✅"):
                    st.success(status)
                elif status.startswith("⚠️"):
                    st.warning(status)
                else:
                    st.error(status)
                st.rerun() 
        if st.button("Refresh Transcript Folder List", key="refresh_folders_btn_download"):
             st.rerun()


    with tab4: # Help
        st.header("How to Use This Application")
        # ### MODIFIED SECTION START ###
        st.markdown(f"""
        ## ⚙️ Initial Setup (LLM Provider & API Keys)
        1.  Go to the **sidebar** (usually on the left).
        2.  Under "**Chat Settings**", choose your preferred **LLM Provider** (OpenAI or Gemini).
        3.  Under "**API Keys**":
            *   Enter your **YouTube Data API v3 Key** (always required for downloading transcripts).
            *   If you selected **OpenAI**: Enter your **OpenAI API Key**.
            *   If you selected **Gemini**: Enter your **Google API Key**.
        4.  These keys can also be set as environment variables (`YOUTUBE_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`). Keys entered in the UI take precedence for the session.

        ## 📥 Downloading Transcripts
        (Same as before)

        ## ⚙️ Creating or Refreshing an Index
        1.  Ensure your desired **LLM Provider** is selected in the sidebar and its API key is entered.
        2.  Go to the "**Setup & Index**" tab.
        3.  Select a transcript folder and click "**Create / Refresh Index**".
            *   The index will be built using the embeddings and LLM of the **currently selected provider**.
            *   **Important**: An index created with OpenAI embeddings is generally not compatible with Gemini embeddings, and vice-versa. If you switch providers, you should ideally re-index your data or load an index previously created with that provider.

        ## ⚙️ Loading an Existing Index
        1.  Ensure the **LLM Provider** selected in the sidebar matches the provider used to create the index you intend to load.
        2.  Go to the "**Setup & Index**" tab.
        3.  Select an index and click "**Load Selected Index**".
            *   The application will attempt to load it using the currently selected provider. Loading may fail if there's a mismatch (e.g., trying to load an OpenAI-based index while Gemini is selected).

        ## 💬 Chatting with Your Videos
        1.  Ensure an index is loaded (check sidebar). The sidebar will show which provider the loaded index is using.
        2.  Go to the "**Chat**" tab and ask questions. The query will use the LLM of the provider associated with the loaded index.

        ## ✨ Tips
        -   When switching LLM providers, it's best practice to create/refresh your index using that new provider or load an index previously built with it.
        -   The `data/storage/your-folder-name/vector/` directory contains the persisted index. If you frequently switch between OpenAI and Gemini for the *same* transcript set, consider using different `folder_name` conventions for their respective indexes to keep them separate (e.g., `my-transcripts-openai` and `my-transcripts-gemini`).
        """)
        # ### MODIFIED SECTION END ###

if __name__ == "__main__":
    run_streamlit_app()