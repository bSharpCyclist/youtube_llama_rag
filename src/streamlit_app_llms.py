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
        if "summary_output" not in st.session_state:
            st.session_state.summary_output = ""
        if "generated_outline" not in st.session_state:
            st.session_state.generated_outline = None
        if "selected_outline_topic" not in st.session_state:
            st.session_state.selected_outline_topic = None
        if "drilldown_content" not in st.session_state:
            st.session_state.drilldown_content = None
        if "current_index_name" not in st.session_state:
            st.session_state.current_index_name = None
        if "youtube_api_key_st" not in st.session_state:
            st.session_state.youtube_api_key_st = self.youtube_api_key
        if "openai_api_key_st" not in st.session_state:
            st.session_state.openai_api_key_st = os.getenv('OPENAI_API_KEY', '')
        if "gemini_api_key_st" not in st.session_state: 
            st.session_state.gemini_api_key_st = os.getenv('GEMINI_API_KEY', '') 
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
        try:
            folders = ["Select an index..."]
            current_provider = st.session_state.llm_provider_st
            storage_root = Path(self.base_storage_dir)

            if storage_root.exists():
                for transcript_set_folder in storage_root.iterdir():
                    if not transcript_set_folder.is_dir():
                        continue
                    
                    provider_base_path = transcript_set_folder / current_provider
                    if provider_base_path.exists() and provider_base_path.is_dir():
                        vector_marker = provider_base_path / 'vector_index' / 'docstore.json'
                        summary_marker = provider_base_path / 'summary_index' / 'docstore.json'
                        if vector_marker.exists() or summary_marker.exists():
                            folders.append(transcript_set_folder.name)
            
            return sorted(list(set(folders))) if len(folders) > 1 else folders
        except Exception as e:
            logger.error(f"Error getting index folders for provider {st.session_state.llm_provider_st}: {str(e)}", exc_info=True)
            return ["Select an index..."]
            
    def _prepare_api_keys_for_rag(self) -> bool:
        provider = st.session_state.llm_provider_st
        api_key_to_check = ""
        env_var_name = ""

        if provider == "openai":
            api_key_to_check = st.session_state.openai_api_key_st
            env_var_name = 'OPENAI_API_KEY'
        elif provider == "gemini":
            api_key_to_check = st.session_state.gemini_api_key_st
            env_var_name = 'GEMINI_API_KEY'
        else:
            st.error(f"Unsupported provider: {provider}")
            return False

        if not api_key_to_check:
            st.error(f"{provider.capitalize()} API Key is not set. Please set it in the 'Settings' sidebar.")
            return False
        
        if os.getenv(env_var_name) != api_key_to_check:
            os.environ[env_var_name] = api_key_to_check
            logger.info(f"Set {env_var_name} from UI for current session.")
        return True

    def create_or_load_rag_instance(self, transcript_folder_name: str, action: str = "load") -> str:
        if not self._prepare_api_keys_for_rag():
            return "⚠️ API Key for the selected LLM provider is missing."

        transcript_dir = os.path.join(self.base_transcript_dir, transcript_folder_name)
        
        try:
            spinner_msg = f"{action.capitalize()}ing RAG indexes for '{transcript_folder_name}' using {st.session_state.llm_provider_st.capitalize()}..."
            with st.spinner(spinner_msg):
                st.session_state.llama_rag_instance = LlamaIndexRAG(
                    transcript_dir=transcript_dir,
                    storage_dir_base=self.base_storage_dir, 
                    llm_provider=st.session_state.llm_provider_st
                )

            if action == "create": # "create" implies refresh
                with st.spinner(f"Refreshing indexes for '{transcript_folder_name}'..."):
                    st.session_state.llama_rag_instance.refresh_indexes()

            if not st.session_state.llama_rag_instance or \
               not (st.session_state.llama_rag_instance.vector_index or st.session_state.llama_rag_instance.summary_index):
                instance_exists_msg = "Instance or indexes not available."
                if st.session_state.llama_rag_instance:
                    instance_exists_msg = (f"VectorIndex: {bool(st.session_state.llama_rag_instance.vector_index)}, "
                                           f"SummaryIndex: {bool(st.session_state.llama_rag_instance.summary_index)}")
                
                logger.error(f"Indexes for '{transcript_folder_name}' ({st.session_state.llm_provider_st}) not available after {action}. {instance_exists_msg}")
                st.session_state.llama_rag_instance = None 
                st.session_state.current_index_name = None
                return f"❌ Failed to {action} indexes for '{transcript_folder_name}'. Check logs and ensure transcript files exist."

            st.session_state.current_index_name = transcript_folder_name 
            stats = st.session_state.llama_rag_instance.get_document_stats()
            action_verb = "created/refreshed" if action == "create" else "loaded"
            status_message = f"✅ Indexes for '{transcript_folder_name}' ({st.session_state.llm_provider_st.capitalize()}) {action_verb}. "
            if stats:
                status_message += f"{stats['document_count']} docs, {stats['node_count']} vector nodes. Ready."
            else:
                status_message += "Ready."
            return status_message
            
        except Exception as e:
            logger.error(f"Error during RAG instance {action} for '{transcript_folder_name}': {str(e)}", exc_info=True)
            st.session_state.llama_rag_instance = None 
            st.session_state.current_index_name = None
            return f"❌ Error {action}ing indexes for '{transcript_folder_name}': {str(e)}"

    def create_indexes_for_folder(self, folder_name: str) -> str:
        if not folder_name or folder_name == "Select a folder...":
            return "⚠️ Please select a transcript folder"
        return self.create_or_load_rag_instance(folder_name, action="create")
            
    def load_indexes_for_folder(self, index_name: str) -> str:
        if not index_name or index_name == "Select an index...":
            return "⚠️ Please select an index"
        return self.create_or_load_rag_instance(index_name, action="load")

    def download_transcripts_from_playlist(self, playlist_url: str, folder_name: str) -> str:
        if not st.session_state.youtube_api_key_st: return "⚠️ YouTube API Key is not set."
        if not playlist_url.strip(): return "⚠️ Please enter a playlist URL or ID"
        if not folder_name.strip(): return "⚠️ Please enter a folder name"
        playlist_match = re.search(r'list=([^&]+)', playlist_url)
        playlist_id = playlist_match.group(1) if playlist_match else playlist_url.strip()
        if not playlist_id: return "⚠️ Could not extract playlist ID"
        output_dir = os.path.join(self.base_transcript_dir, folder_name.strip())
        try:
            with st.spinner(f"Downloading playlist '{playlist_id}'..."):
                c = save_transcripts_from_playlist(st.session_state.youtube_api_key_st, playlist_id, output_dir)
            return f"✅ Downloaded {c} transcripts to '{folder_name}'." if c > 0 else f"⚠️ No transcripts downloaded for playlist '{playlist_id}'."
        except Exception as e: 
            logger.error(f"Error downloading playlist: {e}", exc_info=True)
            return f"❌ Error downloading playlist: {e}"

    def download_transcripts_from_videos(self, video_urls: str, folder_name: str) -> str:
        if not st.session_state.youtube_api_key_st: return "⚠️ YouTube API Key is not set."
        if not video_urls.strip(): return "⚠️ Please enter at least one video URL"
        if not folder_name.strip(): return "⚠️ Please enter a folder name"
        urls = [url.strip() for url in video_urls.split('\n') if url.strip()]
        if not urls: return "⚠️ No valid URLs provided"
        output_dir = os.path.join(self.base_transcript_dir, folder_name.strip())
        try:
            with st.spinner(f"Downloading video transcripts..."):
                c = save_transcripts_from_video_list(urls, output_dir)
            return f"✅ Downloaded {c} transcripts to '{folder_name}'." if c > 0 else "⚠️ No transcripts downloaded from the provided video URLs."
        except Exception as e: 
            logger.error(f"Error downloading videos: {e}", exc_info=True)
            return f"❌ Error downloading videos: {e}"

    def get_sources_text(self, source_nodes) -> str:
        if not source_nodes: return ""
        sources_text = "**Sources:**\n\n"
        for i, source in enumerate(source_nodes):
            title = source.node.metadata.get('episode_title', source.node.metadata.get('file_name', f"Source {i+1}"))
            score = source.score if hasattr(source, 'score') else "N/A" 
            text = source.node.get_content()
            sources_text += f"**Source {i+1}:** {title}  \n"
            if score != "N/A" and isinstance(score, float): sources_text += f"*Relevance Score: {score:.4f}*\n\n"
            else: sources_text += "\n"
            sources_text += f"```text\n{text[:300]}...\n```\n\n" 
        return sources_text
        
    def process_qna_message(self, message: str, top_k: int = 3, show_sources: bool = True) -> Tuple[str, str]:
        if not message.strip(): return "", ""
        rag_instance = st.session_state.llama_rag_instance
        if not rag_instance or not rag_instance.vector_index:
            return "Q&A Index not loaded. Please load an index in 'Setup & Index' tab.", ""
        
        provider = rag_instance.llm_provider 
        api_key_ok = True
        if provider == "openai" and not st.session_state.openai_api_key_st: api_key_ok = False
        elif provider == "gemini" and not st.session_state.gemini_api_key_st: api_key_ok = False
        if not api_key_ok: return f"{provider.capitalize()} API Key not set for LLM. Set in sidebar.", ""
            
        try:
            with st.spinner(f"Thinking with {provider.capitalize()} for Q&A..."):
                response_obj, source_nodes = rag_instance.query_vector_index(
                    message, similarity_top_k=top_k
                )
            sources_text_md = self.get_sources_text(source_nodes) if show_sources else ""
            resp_text = response_obj.response if hasattr(response_obj, 'response') and response_obj.response is not None else "No response generated."
            return resp_text, sources_text_md
        except Exception as e:
            logger.error(f"Error processing Q&A query: {str(e)}", exc_info=True)
            return f"Error processing Q&A query: {str(e)}", ""

    def generate_summary_from_ui(self, summary_query: Optional[str] = None) -> str:
        rag_instance = st.session_state.llama_rag_instance
        if not rag_instance or not rag_instance.summary_index:
            return "Summary Index not loaded. Please load an index in 'Setup & Index' tab."

        provider = rag_instance.llm_provider
        api_key_ok = True
        if provider == "openai" and not st.session_state.openai_api_key_st: api_key_ok = False
        elif provider == "gemini" and not st.session_state.gemini_api_key_st: api_key_ok = False
        if not api_key_ok: return f"{provider.capitalize()} API Key not set for LLM. Set in sidebar."

        try:
            with st.spinner(f"Generating summary with {provider.capitalize()}..."):
                response_obj, _ = rag_instance.query_summary_index(query_text=summary_query, output_format="summary")
            resp_text = response_obj.response if hasattr(response_obj, 'response') and response_obj.response is not None else "No summary generated."
            return resp_text
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            return f"Error generating summary: {str(e)}"

    def generate_outline_from_ui(self, outline_query: Optional[str] = None) -> str:
        rag_instance = st.session_state.llama_rag_instance
        if not rag_instance or not rag_instance.summary_index:
            return "Outline generation requires a Summary Index. Please load/create indexes."

        provider = rag_instance.llm_provider
        api_key_ok = True
        if provider == "openai" and not st.session_state.openai_api_key_st: api_key_ok = False
        elif provider == "gemini" and not st.session_state.gemini_api_key_st: api_key_ok = False
        if not api_key_ok: return f"{provider.capitalize()} API Key not set for LLM. Set in sidebar."

        try:
            with st.spinner(f"Generating outline with {provider.capitalize()}..."):
                response_obj, _ = rag_instance.query_summary_index(
                    query_text=outline_query, 
                    output_format="outline"
                )
            resp_text = response_obj.response if hasattr(response_obj, 'response') and response_obj.response is not None else "No outline generated."
            return resp_text
        except Exception as e:
            logger.error(f"Error generating outline: {str(e)}", exc_info=True)
            return f"Error generating outline: {str(e)}"

def run_streamlit_app():
    st.set_page_config(layout="wide", page_title="YouTube RAG Chatbot")
    st.title("📚 YouTube Transcript RAG Chatbot")
    rag_app = YouTubeRAGStreamlit()

    if "generated_outline" not in st.session_state:
        st.session_state.generated_outline = None
    if "selected_outline_topic" not in st.session_state:
        st.session_state.selected_outline_topic = None
    if "drilldown_content" not in st.session_state:
        st.session_state.drilldown_content = None

    with st.sidebar:
        st.header("LLM Configuration")
        
        def provider_changed_callback():
            logger.info(f"LLM Provider changed to: {st.session_state.llm_provider_selector_sidebar}")
            if "current_index_name" in st.session_state: st.session_state.current_index_name = None
            if "llama_rag_instance" in st.session_state: st.session_state.llama_rag_instance = None
            if "chat_messages" in st.session_state: st.session_state.chat_messages = []
            if "summary_output" in st.session_state: st.session_state.summary_output = ""
            st.session_state.generated_outline = None
            st.session_state.selected_outline_topic = None
            st.session_state.drilldown_content = None

        st.session_state.llm_provider_st = st.selectbox(
            "Choose LLM Provider (for Embeddings & Generation)",
            options=["openai", "gemini"],
            index=["openai", "gemini"].index(st.session_state.get("llm_provider_st", "openai")),
            key="llm_provider_selector_sidebar",
            on_change=provider_changed_callback 
        )
        st.header("Chat Settings")
        top_k_slider = st.slider("Number of sources (for Q&A)", 1, 10, 3, key="top_k_slider_main_sidebar")
        show_sources_checkbox = st.checkbox("Show sources in Q&A", True, key="show_sources_main_sidebar")
        st.header("API Keys")
        st.session_state.youtube_api_key_st = st.text_input("YouTube API Key", value=st.session_state.get("youtube_api_key_st", ""), type="password")
        if not st.session_state.youtube_api_key_st: st.warning("YouTube API Key not set.")
        
        if st.session_state.llm_provider_st == "openai":
            st.session_state.openai_api_key_st = st.text_input("OpenAI API Key", value=st.session_state.get("openai_api_key_st", ""), type="password")
            if not st.session_state.openai_api_key_st: st.warning("OpenAI API Key not set.")
        elif st.session_state.llm_provider_st == "gemini": 
            st.session_state.gemini_api_key_st = st.text_input("Gemini API Key", value=st.session_state.get("gemini_api_key_st", ""), type="password")
            if not st.session_state.gemini_api_key_st: st.warning("Gemini API Key not set.")

        st.header("Loaded Index Status")
        if st.session_state.current_index_name and st.session_state.llama_rag_instance:
            rag_instance = st.session_state.llama_rag_instance
            if rag_instance.llm_provider.lower() == st.session_state.llm_provider_st.lower():
                st.success(f"Active Index: **{st.session_state.current_index_name}**")
                st.markdown(f"Provider: **{rag_instance.llm_provider.capitalize()}**")
                
                # Assuming LlamaIndexRAG has actual_llm_model_name and actual_embedding_model_name
                # If not, these will fall back to preferred names or "Default"
                llm_model_disp = getattr(rag_instance, 'actual_llm_model_name', rag_instance.llm_model_name or "Default")
                embed_model_disp = getattr(rag_instance, 'actual_embedding_model_name', rag_instance.embedding_model_name or "Default")
                st.markdown(f"LLM Model: `{llm_model_disp}`")
                st.markdown(f"Embedding Model: `{embed_model_disp}`")

                stats = rag_instance.get_document_stats()
                if stats: st.caption(f"({stats['document_count']} docs, {stats['node_count']} vector nodes)")
                
                if rag_instance.vector_index: st.caption("Q&A Index: ✅ Available")
                else: st.caption("Q&A Index: ❌ Not Available")
                if rag_instance.summary_index: st.caption("Summary Index: ✅ Available")
                else: st.caption("Summary Index: ❌ Not Available")
            else:
                st.warning(f"Provider Mismatch! UI: {st.session_state.llm_provider_st.capitalize()}, Loaded: {rag_instance.llm_provider.capitalize()}. Load/Create index for selected provider.")
                if st.button("Clear Mismatched Index", key="clear_mismatch_btn_sidebar"):
                    st.session_state.current_index_name = None; st.session_state.llama_rag_instance = None
                    st.rerun()
        else:
            st.info(f"No index loaded for {st.session_state.llm_provider_st.capitalize()}. Use 'Setup & Index' tab.")

    tab_interact, tab_setup, tab_download, tab_help = st.tabs(["💬 Interact", "⚙️ Setup & Index", "📥 Download Transcripts", "❓ Help"])

    with tab_interact: 
        st.header("Interact with Indexed Content")
        
        interaction_mode = st.radio(
            "Choose interaction mode:",
            ("Question & Answer (Q&A)", "Generate Summary", "Explore Outline / Table of Contents"),
            key="interaction_mode_radio_main"
        )

        if interaction_mode == "Question & Answer (Q&A)":
            st.subheader("Ask a Question")
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                        with st.expander("View Sources"): st.markdown(msg["sources"])
            if prompt_qa := st.chat_input("Ask your question...", key="qna_input_main_tab"):
                st.session_state.chat_messages.append({"role": "user", "content": prompt_qa})
                with st.chat_message("user"): st.markdown(prompt_qa)
                with st.chat_message("assistant"):
                    response_text, sources_md = rag_app.process_qna_message(
                        prompt_qa, 
                        st.session_state.top_k_slider_main_sidebar, 
                        st.session_state.show_sources_main_sidebar
                    )
                    st.markdown(response_text)
                    if sources_md:
                        with st.expander("View Sources"): st.markdown(sources_md)
                st.session_state.chat_messages.append({"role": "assistant", "content": response_text, "sources": sources_md})
                st.rerun() 
            if st.session_state.chat_messages and st.button("Clear Q&A History", key="clear_qna_btn_main_tab"):
                st.session_state.chat_messages = []
                st.rerun()

        elif interaction_mode == "Generate Summary":
            st.subheader("Generate Summary")
            summary_query_topic = st.text_input(
                "Optional: Enter a topic to focus the summary on (leave blank for general summary):", 
                key="summary_topic_input_main_tab"
            )
            if st.button("Get Summary", key="get_summary_btn_main_tab"):
                summary_text = rag_app.generate_summary_from_ui(summary_query=summary_query_topic if summary_query_topic.strip() else None)
                st.session_state.summary_output = summary_text
                st.rerun()
            if st.session_state.summary_output:
                st.markdown("### Summary Result:")
                st.info(st.session_state.summary_output)
                if st.button("Clear Summary Output", key="clear_summary_btn_main_tab"):
                    st.session_state.summary_output = ""
                    st.rerun()

        elif interaction_mode == "Explore Outline / Table of Contents":
            st.subheader("Explore Content Outline")
            
            outline_query_topic = st.text_input(
                "Optional: Enter a topic to focus the outline on (leave blank for full outline):", 
                key="outline_topic_input_main_tab"
            )
            if st.button("Generate Outline / Table of Contents", key="generate_outline_btn_main_tab"):
                st.session_state.selected_outline_topic = None 
                st.session_state.drilldown_content = None
                outline_text = rag_app.generate_outline_from_ui(outline_query=outline_query_topic if outline_query_topic.strip() else None)
                st.session_state.generated_outline = outline_text
                st.rerun()

            if st.session_state.generated_outline:
                st.markdown("### Generated Outline / Table of Contents:")
                st.markdown(st.session_state.generated_outline) 
                
                st.divider()
                st.markdown("#### Drill-down into a topic from the outline:")
                drilldown_topic = st.text_input("Enter a topic from the outline to get more details:", key="drilldown_topic_input_main_tab")
                if st.button("Get Details for Topic", key="drilldown_btn_main_tab"):
                    if drilldown_topic.strip():
                        st.session_state.selected_outline_topic = drilldown_topic
                        with st.spinner(f"Fetching details for '{drilldown_topic}'..."):
                            response_text, sources_md = rag_app.process_qna_message(
                                drilldown_topic, 
                                top_k=st.session_state.top_k_slider_main_sidebar,
                                show_sources=True
                            )
                        st.session_state.drilldown_content = {"text": response_text, "sources": sources_md}
                        st.rerun()
                    else:
                        st.warning("Please enter a topic to drill down into.")
            
            if st.session_state.selected_outline_topic and st.session_state.drilldown_content:
                st.markdown(f"### Details for: {st.session_state.selected_outline_topic}")
                st.markdown(st.session_state.drilldown_content["text"])
                if st.session_state.drilldown_content["sources"]:
                    with st.expander("View Sources for Details"):
                        st.markdown(st.session_state.drilldown_content["sources"])

    with tab_setup: 
        st.header(f"Manage RAG Indexes (Provider: {st.session_state.llm_provider_st.capitalize()})")
        st.markdown("This will create/load both a Q&A (Vector) Index and a Summary Index.")
        st.subheader(f"Load Existing Indexes")
        index_folders = rag_app.get_index_folders()
        current_selection_load = st.session_state.current_index_name
        default_load_idx = 0
        if current_selection_load and current_selection_load in index_folders:
            if st.session_state.llama_rag_instance and \
               st.session_state.llama_rag_instance.llm_provider.lower() == st.session_state.llm_provider_st.lower():
                default_load_idx = index_folders.index(current_selection_load)
            else: current_selection_load = None 
        if not current_selection_load and index_folders and index_folders[0] == "Select an index...": default_load_idx = 0
        selected_index_to_load = st.selectbox(
            f"Select transcript set to load indexes for", 
            options=index_folders,
            key=f"load_index_select_setup_{st.session_state.llm_provider_st}", 
            index=default_load_idx
        )
        if st.button("Load Selected Indexes", key="load_idx_btn_setup"):
            if selected_index_to_load != "Select an index...":
                status = rag_app.load_indexes_for_folder(selected_index_to_load)
                if status.startswith("✅"):
                    st.success(status)
                elif status.startswith("⚠️"):
                    st.warning(status)
                else:
                    st.error(status)
                st.rerun() 
            else: st.warning("Please select a transcript set.")
        st.divider()
        st.subheader(f"Create or Refresh Indexes from Transcripts")
        transcript_folders = rag_app.get_transcript_folders()
        selected_transcript_folder = st.selectbox("Select transcript folder to build/refresh indexes for", options=transcript_folders, key="create_index_select_setup")
        if st.button("Create / Refresh Indexes", key="create_idx_btn_setup"):
            if selected_transcript_folder != "Select a folder...":
                status = rag_app.create_indexes_for_folder(selected_transcript_folder)
                if status.startswith("✅"):
                    st.success(status)
                elif status.startswith("⚠️"):
                    st.warning(status)
                else:
                    st.error(status)
                st.rerun()
            else: st.warning("Please select a transcript folder.")
        if st.button("Refresh Folder Lists", key="refresh_folders_btn_setup_tab"): st.rerun()

    with tab_download: 
        st.header("Download Transcripts")
        download_tab1, download_tab2 = st.tabs(["From Playlist", "From Videos"])
        with download_tab1:
            st.subheader("Download from YouTube Playlist")
            playlist_url = st.text_input("YouTube Playlist URL or ID", placeholder="e.g., PL...", key="playlist_url_input_dl_tab")
            playlist_folder_name = st.text_input("Destination Folder Name", placeholder="e.g., my-playlist-transcripts", key="playlist_folder_input_dl_tab")
            if st.button("Download Playlist Transcripts", key="download_playlist_btn_dl_tab"):
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
            video_urls = st.text_area("YouTube Video URLs (one per line)", placeholder="e.g., https://www.youtube.com/watch?v=...", key="video_urls_input_dl_tab", height=150)
            videos_folder_name = st.text_input("Destination Folder Name", placeholder="e.g., my-video-transcripts", key="videos_folder_input_dl_tab")
            if st.button("Download Video Transcripts", key="download_videos_btn_dl_tab"):
                status = rag_app.download_transcripts_from_videos(video_urls, videos_folder_name)
                if status.startswith("✅"):
                    st.success(status)
                elif status.startswith("⚠️"):
                    st.warning(status)
                else:
                    st.error(status)
                st.rerun() 
        if st.button("Refresh Transcript Folder List", key="refresh_folders_btn_download_dl_tab"): st.rerun()

    with tab_help: 
        st.header("How to Use This Application")
        st.markdown(f"""
        ## ⚙️ Initial Setup: LLM Provider & API Keys
        1.  Go to the **sidebar**.
        2.  Under "**LLM Configuration**", choose your **LLM Provider** (OpenAI or Gemini). This provider will be used for both embeddings (for Q&A) and LLM generation (for Q&A and Summaries).
        3.  Under "**API Keys**", enter the API key for your chosen provider, and your YouTube Data API v3 Key.

        ## 📥 Downloading Transcripts
        1.  Navigate to the "**Download Transcripts**" tab.
        2.  Choose either "**From Playlist**" or "**From Videos**".
        3.  Provide the YouTube URL(s)/ID and a **Destination Folder Name**. This name will be used to create a subfolder under `data/transcripts/` (e.g., `data/transcripts/your-folder-name/`).
        4.  Click the "**Download ... Transcripts**" button.
        5.  Status messages will indicate success or failure.

        ## ⚙️ Creating or Refreshing Indexes
        1.  Ensure your **LLM Provider** and API keys are set in the sidebar.
        2.  Go to the "**Setup & Index**" tab.
        3.  Select a transcript folder and click "**Create / Refresh Indexes**".
            *   This will build/update **both** a Q&A (Vector) Index and a Summary Index for the selected transcripts using the chosen LLM Provider.
            *   Indexes are stored in provider-specific paths, e.g., `data/storage/transcript_folder/{st.session_state.llm_provider_st}/vector_index/` and `.../summary_index/`.

        ## ⚙️ Loading Existing Indexes
        1.  Select your **LLM Provider** in the sidebar.
        2.  Go to the "**Setup & Index**" tab. The "Load Existing Indexes" dropdown will show transcript sets that have indexes built with the selected provider.
        3.  Select a transcript set and click "**Load Selected Indexes**". Both Q&A and Summary indexes will be loaded.

        ## 💬 Interacting: Q&A, Summarization, and Outline Exploration
        1.  Ensure indexes are loaded (check sidebar status).
        2.  Go to the "**Interact**" tab.
        3.  Choose your desired interaction mode using the radio buttons:
            *   **Question & Answer (Q&A):** Type your specific question into the chat input.
            *   **Generate Summary:** Optionally provide a topic. Click "Get Summary".
            *   **Explore Outline / Table of Contents:** 
                *   Optionally provide a topic to focus the outline. Click "Generate Outline / Table of Contents".
                *   The outline will be displayed.
                *   To get more details on a specific topic from the outline, type that topic into the "Enter a topic from the outline..." input box and click "Get Details for Topic". This uses the Q&A index.
        """)

if __name__ == "__main__":
    run_streamlit_app()