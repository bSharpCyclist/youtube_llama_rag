"""
Gradio-based web interface for YouTube transcript RAG.
"""

import os
import gradio as gr
import logging
from pathlib import Path
import re
import time
from typing import Dict, List, Tuple, Optional

# Import local modules
from .youtube_utils import save_transcripts_from_playlist, save_transcripts_from_video_list, extract_video_id
from .llama_utils import LlamaIndexRAG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeRAGUI:
    def __init__(self):
        """Initialize the YouTubeRAGUI application."""
        self.llama_rag = None
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY', '')
        
        # Set default paths
        self.base_transcript_dir = "data/transcripts"
        self.base_storage_dir = "data/storage"
        
        # Create directories if they don't exist
        os.makedirs(self.base_transcript_dir, exist_ok=True)
        os.makedirs(self.base_storage_dir, exist_ok=True)
        
        # Chat history
        self.chat_history = []
        
    def get_transcript_folders(self) -> List[str]:
        """Get a list of available transcript folders."""
        try:
            return ["Select a folder..."] + [
                f.name for f in Path(self.base_transcript_dir).iterdir() 
                if f.is_dir() and any(f.glob('*.txt'))
            ]
        except Exception as e:
            logger.error(f"Error getting transcript folders: {str(e)}")
            return ["Select a folder..."]
            
    def get_index_folders(self) -> List[str]:
        """Get a list of available index folders."""
        try:
            folders = ["Select an index..."]
            storage_path = Path(self.base_storage_dir)
            if storage_path.exists():
                for folder in storage_path.iterdir():
                    if folder.is_dir() and (folder / 'vector').exists() and (folder / 'vector' / 'docstore.json').exists():
                        folders.append(folder.name)
            return folders
        except Exception as e:
            logger.error(f"Error getting index folders: {str(e)}")
            return ["Select an index..."]
            
    def download_transcripts_from_playlist(
        self, playlist_url: str, folder_name: str
    ) -> Tuple[str, dict]:
        """Download transcripts from a YouTube playlist."""
        if not playlist_url.strip():
            return "⚠️ Please enter a playlist URL or ID", gr.update()
            
        if not folder_name.strip():
            return "⚠️ Please enter a folder name", gr.update()
            
        # Extract playlist ID from URL if needed
        playlist_match = re.search(r'list=([^&]+)', playlist_url)
        playlist_id = playlist_match.group(1) if playlist_match else playlist_url.strip()
        
        if not playlist_id:
            return "⚠️ Could not extract playlist ID", gr.update()
            
        # Create destination directory
        output_dir = os.path.join(self.base_transcript_dir, folder_name.strip())
        
        try:
            # Download transcripts
            success_count = save_transcripts_from_playlist(self.youtube_api_key, playlist_id, output_dir)
            
            if success_count > 0:
                return f"✅ Successfully downloaded {success_count} transcripts to {folder_name}/", gr.update(choices=self.get_transcript_folders())
            else:
                return "⚠️ No transcripts were downloaded. Check if playlist exists and has videos with transcripts.", gr.update()
        except Exception as e:
            logger.error(f"Error downloading transcripts: {str(e)}")
            return f"❌ Error downloading transcripts: {str(e)}", gr.update()
            
    def download_transcripts_from_videos(
        self, video_urls: str, folder_name: str
    ) -> Tuple[str, dict]:
        """Download transcripts from individual YouTube videos."""
        if not video_urls.strip():
            return "⚠️ Please enter at least one video URL", gr.update()
            
        if not folder_name.strip():
            return "⚠️ Please enter a folder name", gr.update()
            
        # Split and clean video URLs
        urls = [url.strip() for url in video_urls.split('\n') if url.strip()]
        
        if not urls:
            return "⚠️ No valid URLs provided", gr.update()
            
        # Create destination directory
        output_dir = os.path.join(self.base_transcript_dir, folder_name.strip())
        
        try:
            # Download transcripts
            success_count = save_transcripts_from_video_list(urls, output_dir)
            
            if success_count > 0:
                return f"✅ Successfully downloaded {success_count} transcripts to {folder_name}/", gr.update(choices=self.get_transcript_folders())
            else:
                return "⚠️ No transcripts were downloaded. Check if videos exist and have transcripts.", gr.update()
        except Exception as e:
            logger.error(f"Error downloading transcripts: {str(e)}")
            return f"❌ Error downloading transcripts: {str(e)}", gr.update()
            
    def create_index(
        self, folder_name: str, progress=gr.Progress()
    ) -> Tuple[str, dict]:
        """Create or refresh an index for the selected transcript folder."""
        if not folder_name or folder_name == "Select a folder...":
            return "⚠️ Please select a transcript folder", gr.update()
            
        transcript_dir = os.path.join(self.base_transcript_dir, folder_name)
        storage_dir = os.path.join(self.base_storage_dir, folder_name, 'vector')
        
        try:
            progress(0, desc="Initializing...")
            
            # Initialize LlamaIndexRAG with selected folders
            self.llama_rag = LlamaIndexRAG(
                transcript_dir=transcript_dir,
                storage_dir=storage_dir
            )
            
            progress(0.3, desc="Loading documents...")
            
            # Create or refresh index
            self.llama_rag.refresh_index()
            
            progress(0.9, desc="Finalizing index...")
            
            # Get stats
            stats = self.llama_rag.get_document_stats()
            if stats:
                return f"✅ Index created successfully. {stats['document_count']} documents indexed into {stats['node_count']} chunks.", gr.update(choices=self.get_index_folders())
            else:
                return "✅ Index created successfully.", gr.update(choices=self.get_index_folders())
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return f"❌ Error creating index: {str(e)}", gr.update()
            
    def load_index(self, index_name: str, progress=gr.Progress()) -> str:
        """Load the selected index."""
        if not index_name or index_name == "Select an index...":
            return "⚠️ Please select an index"

        progress(0.1, desc="Initializing index loading...")
        transcript_dir = os.path.join(self.base_transcript_dir, index_name)
        storage_dir = os.path.join(self.base_storage_dir, index_name, 'vector')

        try:
            progress(0.3, desc="Loading index data...")
            self.llama_rag = LlamaIndexRAG(
                transcript_dir=transcript_dir,
                storage_dir=storage_dir
            )

            progress(0.7, desc="Fetching document stats...")
            stats = self.llama_rag.get_document_stats()

            progress(1.0, desc="Index loaded successfully.")
            if stats:
                return f"✅ Index '{index_name}' loaded successfully. {stats['document_count']} documents in {stats['node_count']} chunks. Ready to chat!"
            else:
                return f"✅ Index '{index_name}' loaded successfully. Ready to chat!"
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return f"❌ Error loading index: {str(e)}"
            
    def get_sources_text(self, source_nodes) -> str:
        """Format source nodes into a readable string."""
        if not source_nodes:
            return ""
            
        sources_text = "**Sources:**\n\n"
        for i, source in enumerate(source_nodes):
            title = source.node.metadata.get('episode_title', f"Source {i+1}")
            score = source.score
            text = source.node.get_content()
            
            sources_text += f"**Source {i+1}:** {title}  \n"
            sources_text += f"*Relevance Score: {score:.4f}*\n\n"
            sources_text += f"```\n{text[:300]}...\n```\n\n"
            
        return sources_text
        
    def chat(
        self, message: str, history: List[Tuple[str, str]], top_k: int = 3, show_sources: bool = True
    ) -> Tuple[str, List[Tuple[str, str]], str]:
        """Process a chat message and return the response with sources."""
        if not message.strip():
            return "", history, ""
            
        if not self.llama_rag or not self.llama_rag.index:
            return "Please load an index first. Go to the 'Setup & Index' tab to load or create an index.", history, ""
            
        try:
            # Execute query
            response, source_nodes = self.llama_rag.query(message, similarity_top_k=top_k)
            
            # Format sources if requested
            sources_text = self.get_sources_text(source_nodes) if show_sources else ""
            
            # Ensure response.response is str and not None
            resp_text = response.response if response.response is not None else ""
            # Ensure all history tuples are (str, str) with no None
            safe_history = [(str(q), str(a) if a is not None else "") for q, a in history]
            # Add the new entry
            safe_history.append((message, resp_text))
            return resp_text, safe_history, sources_text
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            error_msg = f"Error processing your query: {str(e)}"
            return error_msg, history + [(message, error_msg)], ""
            
    def build_gradio_app(self) -> gr.Blocks:
        """Build the Gradio interface."""
        with gr.Blocks(title="YouTube RAG Chatbot") as app:
            gr.Markdown("# YouTube Transcript RAG Chatbot")
            
            with gr.Tabs() as tabs:
                with gr.Tab("Chat"):
                    with gr.Row():
                        with gr.Column(scale=4):
                            chatbot = gr.Chatbot(
                                height=500,
                                show_label=False,
                                elem_id="chatbot",
                                type="messages"
                            )
                            
                            msg = gr.Textbox(
                                placeholder="Ask a question about the YouTube videos...",
                                container=False,
                                scale=7,
                            )
                            
                            with gr.Row():
                                submit_btn = gr.Button("Submit", variant="primary")
                                clear_btn = gr.Button("Clear Chat")
                                
                        with gr.Column(scale=2):
                            gr.Markdown("## Sources")
                            sources_display = gr.Markdown(height=500)
                            
                            with gr.Accordion("Query Settings", open=False):
                                top_k = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    step=1,
                                    value=3,
                                    label="Number of sources to retrieve (top_k)"
                                )
                                show_sources = gr.Checkbox(
                                    value=True,
                                    label="Show sources"
                                )
                                
                with gr.Tab("Setup & Index"):
                    gr.Markdown("## Load or Create Index")
                    
                    with gr.Row():
                        with gr.Column():
                            index_dropdown = gr.Dropdown(
                                choices=self.get_index_folders(),
                                label="Select an index to load",
                                interactive=True,
                            )
                            
                            load_index_btn = gr.Button("Load Selected Index", variant="primary")
                            index_status = gr.Markdown()
                            
                    gr.Markdown("## Create New Index")
                    
                    with gr.Row():
                        with gr.Column():
                            transcript_folder_dropdown = gr.Dropdown(
                                choices=self.get_transcript_folders(),
                                label="Select transcript folder",
                                interactive=True,
                            )
                            
                            refresh_folder_btn = gr.Button("Refresh Folder List")
                            create_index_btn = gr.Button("Create/Refresh Index", variant="primary")
                            new_index_status = gr.Markdown()
                            
                with gr.Tab("Download Transcripts"):
                    with gr.Tabs() as download_tabs:
                        with gr.Tab("From Playlist"):
                            playlist_url = gr.Textbox(
                                label="YouTube Playlist URL or ID",
                                placeholder="https://www.youtube.com/playlist?list=PLAYLIST_ID or just the PLAYLIST_ID"
                            )
                            
                            playlist_folder_name = gr.Textbox(
                                label="Destination Folder Name",
                                placeholder="e.g., ancient-aliens"
                            )
                            
                            playlist_download_btn = gr.Button("Download Transcripts", variant="primary")
                            playlist_status = gr.Markdown()
                            
                        with gr.Tab("From Videos"):
                            video_urls = gr.TextArea(
                                label="YouTube Video URLs (one per line)",
                                placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=..."
                            )
                            
                            videos_folder_name = gr.Textbox(
                                label="Destination Folder Name",
                                placeholder="e.g., ted-talks"
                            )
                            
                            videos_download_btn = gr.Button("Download Transcripts", variant="primary")
                            videos_status = gr.Markdown()
                            
                with gr.Tab("Help"):
                    gr.Markdown("""
                    # How to Use This Application
                    
                    ## Downloading Transcripts
                    1. Go to the "Download Transcripts" tab
                    2. Choose either "From Playlist" or "From Videos"
                    3. Enter the playlist URL/ID or individual video URLs
                    4. Provide a folder name for saving the transcripts
                    5. Click "Download Transcripts"
                    
                    ## Creating an Index
                    1. Go to the "Setup & Index" tab
                    2. Select a transcript folder from the dropdown
                    3. Click "Create/Refresh Index"
                    4. Wait for the indexing process to complete
                    
                    ## Loading an Existing Index
                    1. Go to the "Setup & Index" tab
                    2. Select an index from the dropdown
                    3. Click "Load Selected Index"
                    
                    ## Chatting with Your Videos
                    1. Go to the "Chat" tab
                    2. Type your question in the text input
                    3. Click "Submit" or press Enter
                    4. View the AI response and the source information
                    
                    ## Tips
                    - More specific questions tend to get better answers
                    - You can adjust the number of sources retrieved using the slider
                    - The "Show sources" checkbox toggles source display
                    """)
                    
            # Set up event handlers
            
            # Chat tab
            submit_btn.click(
                self.chat,
                inputs=[msg, chatbot, top_k, show_sources],
                outputs=[msg, chatbot, sources_display],
            )
            
            msg.submit(
                self.chat,
                inputs=[msg, chatbot, top_k, show_sources],
                outputs=[msg, chatbot, sources_display],
            )
            
            clear_btn.click(lambda: (None, [], ""), outputs=[msg, chatbot, sources_display])
            
            # Setup & Index tab
            load_index_btn.click(
                self.load_index,
                inputs=[index_dropdown],
                outputs=[index_status],
                show_progress="full"
            )
            
            refresh_folder_btn.click(
                lambda: gr.update(choices=self.get_transcript_folders()),
                outputs=[transcript_folder_dropdown]
            )
            
            create_index_btn.click(
                self.create_index,
                inputs=[transcript_folder_dropdown],
                outputs=[new_index_status, index_dropdown],
                show_progress="full"  # Add this line
            )
            
            # Download tab
            playlist_download_btn.click(
                self.download_transcripts_from_playlist,
                inputs=[playlist_url, playlist_folder_name],
                outputs=[playlist_status, transcript_folder_dropdown]
            )
            
            videos_download_btn.click(
                self.download_transcripts_from_videos,
                inputs=[video_urls, videos_folder_name],
                outputs=[videos_status, transcript_folder_dropdown]
            )
            
        return app
        
def run_ui():
    """Run the Gradio UI application."""
    ui = YouTubeRAGUI()
    app = ui.build_gradio_app()
    return app
    
if __name__ == "__main__":
    app = run_ui()
    app.launch()