# streamlit_app.py
import streamlit as st
import os
import logging
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

try:
    from youtube_utils import save_transcripts_from_playlist, save_transcripts_from_video_list
    from llama_utils_llms import LlamaIndexRAG # Assuming llama_utils.py is updated as per previous response
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
        self.base_storage_dir = "data/storage" # This is the root for all provider storages
        
        os.makedirs(self.base_transcript_dir, exist_ok=True)
        os.makedirs(self.base_storage_dir, exist_ok=True)

        if "llama_rag_instance" not in st.session_state:
            st.session_state.llama_rag_instance = None
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [] 
        if "current_index_name" not in st.session_state: # This will now be just the 'folder_name' part
            st.session_state.current_index_name = None
        if "youtube_api_key_st" not in st.session_state:
            st.session_state.youtube_api_key_st = self.youtube_api_key
        if "openai_api_key_st" not in st.session_state:
            st.session_state.openai_api_key_st = os.getenv('OPENAI_API_KEY', '')
        if "google_api_key_st" not in st.session_state: 
            st.session_state.google_api_key_st = os.getenv('GOOGLE_API_KEY', '') 
        if "llm_provider_st" not in st.session_state: 
            st.session_state.llm_provider_st = "openai"


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
        """Get a list of available index folders (transcript set names)
           that have an index for the CURRENTLY SELECTED LLM PROVIDER."""
        try:
            folders = ["Select an index..."]
            current_provider = st.session_state.llm_provider_st
            
            storage_root = Path(self.base_storage_dir)
            if storage_root.exists():
                for transcript_set_folder in storage_root.iterdir():
                    if not transcript_set_folder.is_dir():
                        continue
                    
                    provider_specific_index_base = transcript_set_folder / current_provider / 'vector'
                    
                    if provider_specific_index_base.exists() and \
                       (provider_specific_index_base / 'docstore.json').exists():
                        folders.append(transcript_set_folder.name)
            
            return sorted(list(set(folders))) if len(folders) > 1 else folders
        except Exception as e:
            logger.error(f"Error getting index folders for provider {st.session_state.llm_provider_st}: {str(e)}", exc_info=True)
            return ["Select an index..."]
            
    def _prepare_api_keys_for_rag(self) -> bool:
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
        
        if not self._prepare_api_keys_for_rag():
            return "⚠️ API Key for the selected LLM provider is missing."

        transcript_dir = os.path.join(self.base_transcript_dir, folder_name)
        
        status_message = ""
        try:
            with st.spinner(f"Initializing RAG for '{folder_name}' using {st.session_state.llm_provider_st}..."):
                st.session_state.llama_rag_instance = LlamaIndexRAG(
                    transcript_dir=transcript_dir,
                    storage_dir_base=self.base_storage_dir, 
                    llm_provider=st.session_state.llm_provider_st
                )

            if not st.session_state.llama_rag_instance or not st.session_state.llama_rag_instance.index:
                logger.warning(f"Index for '{folder_name}' ({st.session_state.llm_provider_st}) not loaded/created on init. Attempting refresh/creation.")

            with st.spinner(f"Refreshing index for '{folder_name}' ({st.session_state.llm_provider_st})..."):
                st.session_state.llama_rag_instance.refresh_index() 

            if not st.session_state.llama_rag_instance.index:
                 logger.error(f"Index for '{folder_name}' ({st.session_state.llm_provider_st}) still not available after refresh.")
                 st.session_state.llama_rag_instance = None 
                 st.session_state.current_index_name = None
                 return f"❌ Failed to create or refresh index for '{folder_name}' ({st.session_state.llm_provider_st}). Ensure transcript files exist and API keys are correct."

            st.session_state.current_index_name = folder_name 
            stats = st.session_state.llama_rag_instance.get_document_stats()
            status_message = f"✅ Index for '{folder_name}' ({st.session_state.llm_provider_st}) "
            if stats:
                status_message += f"created/refreshed. {stats['document_count']} docs, {stats['node_count']} chunks. Now active."
            else:
                status_message += "created/refreshed. Now active."
            
        except Exception as e:
            logger.error(f"Error creating/refreshing index for '{folder_name}' ({st.session_state.llm_provider_st}): {str(e)}", exc_info=True)
            st.session_state.llama_rag_instance = None 
            st.session_state.current_index_name = None
            status_message = f"❌ Error creating/refreshing index for '{folder_name}' ({st.session_state.llm_provider_st}): {str(e)}"
        return status_message
            
    def load_index(self, index_name: str) -> str:
        if not index_name or index_name == "Select an index...":
            return "⚠️ Please select an index"
        
        if not self._prepare_api_keys_for_rag():
            return "⚠️ API Key for the selected LLM provider is missing."

        transcript_dir = os.path.join(self.base_transcript_dir, index_name)

        try:
            with st.spinner(f"Loading index '{index_name}' using {st.session_state.llm_provider_st}..."):
                st.session_state.llama_rag_instance = LlamaIndexRAG(
                    transcript_dir=transcript_dir, 
                    storage_dir_base=self.base_storage_dir,
                    llm_provider=st.session_state.llm_provider_st
                )

                if not st.session_state.llama_rag_instance.index:
                    st.session_state.llama_rag_instance = None 
                    st.session_state.current_index_name = None
                    logger.error(f"Failed to load index '{index_name}' for provider {st.session_state.llm_provider_st}.")
                    return (f"❌ Failed to load index '{index_name}' for {st.session_state.llm_provider_st}. "
                            f"Ensure an index was previously created for this provider and transcript set.")

                stats = st.session_state.llama_rag_instance.get_document_stats()
            
            st.session_state.current_index_name = index_name
            status_message = f"✅ Index '{index_name}' ({st.session_state.llm_provider_st}) loaded. "
            if stats:
                status_message += f"{stats['document_count']} docs, {stats['node_count']} chunks. Ready to chat!"
            else:
                status_message += "Ready to chat!"
            return status_message
        except Exception as e:
            logger.error(f"Error loading index '{index_name}' ({st.session_state.llm_provider_st}): {str(e)}", exc_info=True)
            st.session_state.llama_rag_instance = None
            st.session_state.current_index_name = None
            return f"❌ Error loading index '{index_name}' ({st.session_state.llm_provider_st}): {str(e)}"

    def download_transcripts_from_playlist(self, playlist_url: str, folder_name: str) -> str:
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

    def download_transcripts_from_videos(self, video_urls: str, folder_name: str) -> str:
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
            with st.spinner(f"Downloading transcripts for videos to {folder_name}..."):
                success_count = save_transcripts_from_video_list(urls, output_dir)
            
            if success_count > 0:
                return f"✅ Successfully downloaded {success_count} transcripts to '{folder_name}/'"
            else:
                return "⚠️ No transcripts were downloaded. Check if videos exist and have transcripts."
        except Exception as e:
            logger.error(f"Error downloading video transcripts: {str(e)}", exc_info=True)
            return f"❌ Error downloading transcripts: {str(e)}"

    def get_sources_text(self, source_nodes) -> str:
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
        
        provider = st.session_state.llm_provider_st
        api_key_missing_msg = ""
        if provider == "openai" and not st.session_state.openai_api_key_st:
            api_key_missing_msg = "OpenAI API Key is not set. Cannot query."
        elif provider == "gemini" and not st.session_state.google_api_key_st:
            api_key_missing_msg = "Google API Key is not set for Gemini. Cannot query."
        
        if api_key_missing_msg:
            return f"{api_key_missing_msg} Please set it in 'Settings' sidebar.", ""
            
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
        
        def provider_changed_callback():
            # This callback is triggered when the selectbox value changes.
            # Streamlit automatically reruns the script.
            # We can add logic here if needed, e.g., to clear current_index_name,
            # but the filtering in get_index_folders and display logic should handle it.
            logger.info(f"LLM Provider changed to: {st.session_state.llm_provider_selector}")


        st.session_state.llm_provider_st = st.selectbox(
            "Choose LLM Provider",
            options=["openai", "gemini"],
            index=["openai", "gemini"].index(st.session_state.get("llm_provider_st", "openai")),
            key="llm_provider_selector", # This key is used to access the value
            on_change=provider_changed_callback 
        )

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
        
        if st.session_state.llm_provider_st == "openai":
            st.session_state.openai_api_key_st = st.text_input(
                "OpenAI API Key",
                value=st.session_state.get("openai_api_key_st", ""),
                type="password",
                help="Required if OpenAI is selected as LLM Provider."
            )
            if not st.session_state.openai_api_key_st: # Check only if provider is openai
                st.warning("OpenAI API Key not set.")
        
        elif st.session_state.llm_provider_st == "gemini": 
            st.session_state.google_api_key_st = st.text_input( 
                "Google API Key (for Gemini)",
                value=st.session_state.get("google_api_key_st", ""),
                type="password",
                help="Required if Gemini is selected as LLM Provider."
            )
            if not st.session_state.google_api_key_st: # Check only if provider is gemini
                st.warning("Google API Key for Gemini not set.")

        st.header("Loaded Index")
        if st.session_state.current_index_name and st.session_state.llama_rag_instance:
            instance_provider = st.session_state.llama_rag_instance.llm_provider
            if instance_provider.lower() == st.session_state.llm_provider_st.lower():
                st.success(f"Active Index: **{st.session_state.current_index_name}** (Provider: {instance_provider.capitalize()})")
                if st.session_state.llama_rag_instance.index:
                    stats = st.session_state.llama_rag_instance.get_document_stats()
                    if stats:
                        st.caption(f"({stats['document_count']} docs, {stats['node_count']} chunks)")
            else:
                st.warning(f"Provider mismatch! UI shows {st.session_state.llm_provider_st.capitalize()} but last loaded index was for {instance_provider.capitalize()}. Please load/create an index for the selected provider.")
                # Clear the mismatched loaded index to avoid confusion
                st.session_state.current_index_name = None
                st.session_state.llama_rag_instance = None
                st.rerun() # Rerun to reflect the cleared state
        else:
            st.info(f"No index loaded for {st.session_state.llm_provider_st.capitalize()}. Go to 'Setup & Index' tab.")

    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "⚙️ Setup & Index", "📥 Download Transcripts", "❓ Help"])

    with tab1: 
        st.header("Chat with Your YouTube Index")
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                    with st.expander("View Sources"):
                        st.markdown(msg["sources"])
        if prompt := st.chat_input("Ask a question about the indexed videos...", key="chat_input_main"):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                response_text, sources_md = rag_app.process_chat_message(prompt, st.session_state.top_k_slider_main, st.session_state.show_sources_main)
                st.markdown(response_text)
                if sources_md:
                    with st.expander("View Sources"): st.markdown(sources_md)
            st.session_state.chat_messages.append({"role": "assistant", "content": response_text, "sources": sources_md})
            st.rerun() 
        if st.session_state.chat_messages and st.button("Clear Chat History", key="clear_chat_btn"):
            st.session_state.chat_messages = []
            st.rerun()

    with tab2: 
        st.header(f"Manage Your RAG Index (Using: {st.session_state.llm_provider_st.capitalize()})")
        
        st.subheader(f"Load Existing Index (for {st.session_state.llm_provider_st.capitalize()})")
        index_folders = rag_app.get_index_folders() 
        
        current_selection_load = st.session_state.current_index_name
        default_load_idx = 0
        if current_selection_load and current_selection_load in index_folders: # Check if current index is valid for current provider
            default_load_idx = index_folders.index(current_selection_load)
        elif index_folders and index_folders[0] == "Select an index...": # If not, and placeholder exists
             default_load_idx = 0
        
        selected_index_to_load = st.selectbox(
            f"Select an index to load for {st.session_state.llm_provider_st.capitalize()}", 
            options=index_folders,
            key=f"load_index_select_{st.session_state.llm_provider_st}", 
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
        st.subheader(f"Create or Refresh Index from Transcripts (for {st.session_state.llm_provider_st.capitalize()})")
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

    with tab3: 
        st.header("Download Transcripts")
        download_tab1, download_tab2 = st.tabs(["From Playlist", "From Videos"])
        with download_tab1:
            st.subheader("Download from YouTube Playlist")
            playlist_url = st.text_input("YouTube Playlist URL or ID", placeholder="e.g., PL...", key="playlist_url_input")
            playlist_folder_name = st.text_input("Destination Folder Name (for transcripts)", placeholder="e.g., my-playlist-transcripts", key="playlist_folder_input")
            if st.button("Download Playlist Transcripts", key="download_playlist_btn"):
                status = rag_app.download_transcripts_from_playlist(playlist_url, playlist_folder_name)
                if status.startswith("✅"): st.success(status)
                elif status.startswith("⚠️"): st.warning(status)
                else: st.error(status)
                st.rerun() 
        with download_tab2:
            st.subheader("Download from Individual YouTube Videos")
            video_urls = st.text_area("YouTube Video URLs (one per line)", placeholder="e.g., https://www.youtube.com/watch?v=...", key="video_urls_input", height=150)
            videos_folder_name = st.text_input("Destination Folder Name (for transcripts)", placeholder="e.g., my-video-transcripts", key="videos_folder_input")
            if st.button("Download Video Transcripts", key="download_videos_btn"):
                status = rag_app.download_transcripts_from_videos(video_urls, videos_folder_name)
                if status.startswith("✅"): st.success(status)
                elif status.startswith("⚠️"): st.warning(status)
                else: st.error(status)
                st.rerun() 
        if st.button("Refresh Transcript Folder List", key="refresh_folders_btn_download"):
             st.rerun()

    with tab4: 
        st.header("How to Use This Application")
        st.markdown(f"""
        ## ⚙️ Initial Setup (LLM Provider & API Keys)
        1.  Go to the **sidebar** (usually on the left).
        2.  Under "**Chat Settings**", choose your preferred **LLM Provider** (OpenAI or Gemini) using the dropdown.
        3.  Under "**API Keys**":
            *   Enter your **YouTube Data API v3 Key**. This is always required for downloading new transcripts.
            *   If you selected **OpenAI** as the LLM Provider: Enter your **OpenAI API Key**.
            *   If you selected **Gemini** as the LLM Provider: Enter your **Google API Key**.
        4.  These API keys can also be set as environment variables (`YOUTUBE_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`) before running the app. Keys entered in the UI will take precedence for the current session if an environment variable is also set.

        ## 📥 Downloading Transcripts
        1.  Navigate to the "**Download Transcripts**" tab.
        2.  Choose either "**From Playlist**" or "**From Videos**".
        3.  Provide the YouTube URL(s)/ID and a **Destination Folder Name**. This name will be used to create a subfolder under `data/transcripts/` (e.g., `data/transcripts/your-folder-name/`).
        4.  Click the "**Download ... Transcripts**" button.
        5.  Status messages will indicate success or failure.

        ## ⚙️ Creating or Refreshing an Index
        1.  First, ensure your desired **LLM Provider** (OpenAI or Gemini) is selected in the sidebar and its corresponding API key is correctly entered.
        2.  Go to the "**Setup & Index**" tab.
        3.  From the "**Select transcript folder to index/refresh**" dropdown, choose the folder containing the transcripts you want to process.
        4.  Click the "**Create / Refresh Index**" button.
            *   The index will be built (or updated) using the embeddings and LLM of the **currently selected provider** (shown in the tab header).
            *   The index data will be stored in a provider-specific path. For example, if your transcript folder is `my_videos` and you're using Gemini, the index will be in `data/storage/my_videos/gemini/vector/`.
            *   This structure allows an OpenAI index and a Gemini index for the same set of transcripts to exist simultaneously in separate subfolders.
            *   After creation/refresh, this index automatically becomes the active one for chatting.

        ## ⚙️ Loading an Existing Index
        1.  In the sidebar, ensure the **LLM Provider** selected matches the provider that was used to create the index you intend to load.
        2.  Go to the "**Setup & Index**" tab. The dropdown list under "**Load Existing Index**" will automatically update to show **only those indexes that exist for the currently selected provider**.
        3.  Select an index name (this corresponds to your transcript folder name) from this filtered list.
        4.  Click the "**Load Selected Index**" button. This makes the selected index active for chatting.
        5.  The sidebar will confirm the loaded index and its provider. If you change the LLM Provider in the sidebar *after* loading an index, the sidebar will warn you of a mismatch, and you should load an appropriate index for the new provider.

        ## 💬 Chatting with Your Videos
        1.  Ensure an index is loaded and active (check the sidebar status). The chat will use the LLM of the provider associated with the currently loaded index.
        2.  Go to the "**Chat**" tab.
        3.  Type your question about the content of the indexed videos in the input box at the bottom and press Enter or click the send button.
        4.  The AI's response will appear. You can adjust chat settings (number of sources, show/hide sources) in the sidebar.

        ## ✨ Tips
        -   When you switch the **LLM Provider** in the sidebar, the list of available indexes to load (in the "Setup & Index" tab) will automatically update.
        -   If you have transcripts in `data/transcripts/my_channel/` and index them with OpenAI, the index will be at `data/storage/my_channel/openai/vector/`. If you then switch the provider to Gemini in the sidebar and index the same `my_channel` transcripts, that new Gemini-based index will be created at `data/storage/my_channel/gemini/vector/`. Both can exist, and you can switch between them by changing the provider and loading the respective index.
        -   If dropdown lists for transcript folders or indexes don't seem up-to-date after a download or index operation, use the "**Refresh Folder Lists**" button on the "Setup & Index" tab or the "**Refresh Transcript Folder List**" on the "Download Transcripts" tab.
        """)

if __name__ == "__main__":
    run_streamlit_app()