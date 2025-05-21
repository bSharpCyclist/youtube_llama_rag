"""
Panel-based CLI application for YouTube transcript RAG.
"""

import os
import panel as pn
import logging
from typing import Dict, List, Any, Optional, Union
import re
from pathlib import Path

# Import local modules
from .youtube_utils import save_transcripts_from_playlist, save_transcripts_from_video_list, extract_video_id
from .llama_utils import LlamaIndexRAG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Panel
pn.extension()

class YouTubeRAGCLI:
    def __init__(self):
        """Initialize the YouTubeRAGCLI application."""
        self.llama_rag = None
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY', '')
        
        # Set default paths
        self.base_transcript_dir = "data/transcripts"
        self.base_storage_dir = "data/storage"
        
        # Create directories if they don't exist
        os.makedirs(self.base_transcript_dir, exist_ok=True)
        os.makedirs(self.base_storage_dir, exist_ok=True)
        
        # Initialize the UI components
        self._init_ui()
        
    def _init_ui(self):
        """Initialize Panel UI components."""
        # Header
        self.header = pn.pane.Markdown("# YouTube Transcript RAG CLI", css_classes=['text-center'])
        
        # Download Tab UI components
        self.download_title = pn.pane.Markdown("## Download YouTube Transcripts")
        self.download_type_selector = pn.widgets.RadioButtonGroup(
            options=['YouTube Playlist', 'Individual Videos'],
            value='YouTube Playlist',
            name='Download Type'
        )
        self.playlist_id_input = pn.widgets.TextInput(
            name='YouTube Playlist ID',
            placeholder='e.g., PLob1mZcVWOaiVxrCiEyYXcAbmx7UY8ggW',
            width=400
        )
        self.video_urls_input = pn.widgets.TextAreaInput(
            name='YouTube Video URLs (one per line)',
            placeholder='https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=...',
            width=400,
            height=150,
            visible=False
        )
        self.download_folder_input = pn.widgets.TextInput(
            name='Destination Folder Name',
            placeholder='e.g., ancient-aliens',
            value='',
            width=400
        )
        self.download_button = pn.widgets.Button(name='Download Transcripts', button_type='primary')
        self.download_status = pn.pane.Markdown("", width=600)
        
        # Index Tab UI components
        self.index_title = pn.pane.Markdown("## Create/Refresh Index")
        self.available_folders = pn.widgets.Select(
            name='Select Transcript Folder',
            options=self._get_transcript_folders(),
            width=400
        )
        self.refresh_folders_button = pn.widgets.Button(name='↻ Refresh List', button_type='default', width=100)
        self.index_button = pn.widgets.Button(name='Create/Refresh Index', button_type='primary')
        self.index_status = pn.pane.Markdown("", width=600)
        
        # Query Tab UI components
        self.query_title = pn.pane.Markdown("## Query Indexed Transcripts")
        self.index_selector = pn.widgets.Select(
            name='Select Index to Query',
            options=self._get_index_folders(),
            width=400
        )
        self.refresh_indices_button = pn.widgets.Button(name='↻ Refresh List', button_type='default', width=100)
        self.load_index_button = pn.widgets.Button(name='Load Selected Index', button_type='primary')
        self.top_k_slider = pn.widgets.IntSlider(name='Number of sources to retrieve (top_k)', start=1, end=10, value=2, step=1, width=400)
        self.debug_checkbox = pn.widgets.Checkbox(name='Enable debug info', value=False)
        self.query_input = pn.widgets.TextAreaInput(
            name='Enter your question',
            placeholder='e.g., What are the main theories about the construction of the Great Pyramid?',
            height=100,
            width=600
        )
        self.query_button = pn.widgets.Button(name='Submit Query', button_type='primary')
        self.query_result = pn.pane.Markdown("", width=600, sizing_mode='stretch_width')
        self.sources_title = pn.pane.Markdown("### Sources", visible=False)
        self.sources_display = pn.Column(width=600, sizing_mode='stretch_width')
        
        # Help Tab UI components
        self.help_title = pn.pane.Markdown("## Help & Information")
        self.help_content = pn.pane.Markdown("""
        ### Using This Application
        
        #### Download Transcripts
        1. Choose to download from a playlist or individual videos
        2. Enter the playlist ID or paste video URLs
        3. Provide a folder name to save transcripts
        4. Click "Download Transcripts"
        
        #### Create/Refresh Index
        1. Select a transcript folder from the dropdown
        2. Click "Create/Refresh Index" to build a searchable index
        
        #### Query Index
        1. Select an index from the dropdown
        2. Click "Load Selected Index"
        3. Adjust the number of sources to retrieve if desired
        4. Type your question
        5. Click "Submit Query"
        
        #### Tips
        - For playlists, you need a YouTube API key in your .env file
        - You can find the playlist ID in the YouTube URL: `https://www.youtube.com/playlist?list=PLAYLIST_ID`
        - More specific questions usually get better answers
        """)
        
        # Connect UI event handlers
        self.download_type_selector.param.watch(self._toggle_download_inputs, 'value')
        self.download_button.on_click(self._handle_download)
        self.refresh_folders_button.on_click(self._refresh_folder_lists)
        self.index_button.on_click(self._handle_index_creation)
        self.refresh_indices_button.on_click(self._refresh_folder_lists)
        self.load_index_button.on_click(self._load_selected_index)
        self.query_button.on_click(self._handle_query)
        
        # Create the main panel tabs
        self.tabs = pn.Tabs(
            ("Download", self._create_download_tab()),
            ("Index", self._create_index_tab()),
            ("Query", self._create_query_tab()),
            ("Help", self._create_help_tab()),
            sizing_mode='stretch_width'
        )
        
        # Create the main layout
        self.main_layout = pn.Column(
            self.header,
            self.tabs,
            sizing_mode='stretch_width'
        )
        
    def _create_download_tab(self):
        """Create the download tab layout."""
        return pn.Column(
            self.download_title,
            pn.Row(self.download_type_selector),
            self.playlist_id_input,
            self.video_urls_input,
            self.download_folder_input,
            self.download_button,
            self.download_status
        )
        
    def _create_index_tab(self):
        """Create the index tab layout."""
        return pn.Column(
            self.index_title,
            pn.Row(self.available_folders, self.refresh_folders_button),
            self.index_button,
            self.index_status
        )
        
    def _create_query_tab(self):
        """Create the query tab layout."""
        return pn.Column(
            self.query_title,
            pn.Row(self.index_selector, self.refresh_indices_button),
            self.load_index_button,
            self.top_k_slider,
            self.debug_checkbox,
            self.query_input,
            self.query_button,
            self.query_result,
            self.sources_title,
            self.sources_display
        )
        
    def _create_help_tab(self):
        """Create the help tab layout."""
        return pn.Column(
            self.help_title,
            self.help_content
        )
        
    def _toggle_download_inputs(self, event):
        """Toggle visibility of download inputs based on selection."""
        if event.new == 'YouTube Playlist':
            self.playlist_id_input.visible = True
            self.video_urls_input.visible = False
        else:
            self.playlist_id_input.visible = False
            self.video_urls_input.visible = True
            
    def _get_transcript_folders(self):
        """Get list of transcript folders."""
        try:
            folders = [f.name for f in Path(self.base_transcript_dir).iterdir() if f.is_dir()]
            return [''] + sorted(folders)
        except Exception as e:
            logger.error(f"Error getting transcript folders: {str(e)}")
            return ['']
            
    def _get_index_folders(self):
        """Get list of index folders."""
        try:
            folders = []
            storage_path = Path(self.base_storage_dir)
            if storage_path.exists():
                for folder in storage_path.iterdir():
                    if folder.is_dir() and (folder / 'vector').exists() and (folder / 'vector' / 'docstore.json').exists():
                        folders.append(folder.name)
            return [''] + sorted(folders)
        except Exception as e:
            logger.error(f"Error getting index folders: {str(e)}")
            return ['']
            
    def _refresh_folder_lists(self, event=None):
        """Refresh the folder lists."""
        self.available_folders.options = self._get_transcript_folders()
        self.index_selector.options = self._get_index_folders()
        
    def _handle_download(self, event):
        """Handle transcript download."""
        download_type = self.download_type_selector.value
        folder_name = self.download_folder_input.value.strip()
        
        if not folder_name:
            self.download_status.object = "⚠️ Please enter a folder name"
            return
            
        # Create destination directory path
        output_dir = os.path.join(self.base_transcript_dir, folder_name)
        
        if download_type == 'YouTube Playlist':
            playlist_id = self.playlist_id_input.value.strip()
            
            if not playlist_id:
                self.download_status.object = "⚠️ Please enter a playlist ID"
                return
                
            # Extract playlist ID if a full URL was pasted
            playlist_match = re.search(r'list=([^&]+)', playlist_id)
            if playlist_match:
                playlist_id = playlist_match.group(1)
                
            self.download_status.object = f"⏳ Downloading transcripts from playlist {playlist_id}..."
            
            # Call the download function
            success_count = save_transcripts_from_playlist(self.youtube_api_key, playlist_id, output_dir)
            
            if success_count > 0:
                self.download_status.object = f"✅ Successfully downloaded {success_count} transcripts to {folder_name}/"
            else:
                self.download_status.object = f"⚠️ No transcripts were downloaded. Check if playlist exists and has videos with transcripts."
                
        else:  # Individual videos
            video_urls_raw = self.video_urls_input.value
            if video_urls_raw is None or str(video_urls_raw).strip() == "":
                self.download_status.object = "⚠️ Please enter at least one YouTube URL"
                return
            video_urls = str(video_urls_raw).strip().split('\n')
            video_urls = [url.strip() for url in video_urls if url.strip()]
            
            if not video_urls:
                self.download_status.object = "⚠️ Please enter at least one YouTube URL"
                return
                
            self.download_status.object = f"⏳ Downloading transcripts from {len(video_urls)} videos..."
            
            # Call the download function
            success_count = save_transcripts_from_video_list(video_urls, output_dir)
            
            if success_count > 0:
                self.download_status.object = f"✅ Successfully downloaded {success_count} transcripts to {folder_name}/"
            else:
                self.download_status.object = f"⚠️ No transcripts were downloaded. Check if videos exist and have transcripts."
                
        # Refresh folder lists
        self._refresh_folder_lists()
        
    def _handle_index_creation(self, event):
        """Handle index creation/refreshing."""
        folder_name = self.available_folders.value
        
        if not folder_name:
            self.index_status.object = "⚠️ Please select a transcript folder"
            return
            
        transcript_dir = os.path.join(self.base_transcript_dir, folder_name)
        storage_dir = os.path.join(self.base_storage_dir, folder_name, 'vector')
        
        self.index_status.object = f"⏳ Creating/refreshing index for {folder_name}..."
        
        try:
            # Initialize LlamaIndexRAG with the selected folders
            self.llama_rag = LlamaIndexRAG(
                transcript_dir=transcript_dir,
                storage_dir=storage_dir
            )
            
            # Refresh the index
            self.llama_rag.refresh_index()
            
            # Get stats
            stats = self.llama_rag.get_document_stats()
            if stats:
                self.index_status.object = f"✅ Index created/refreshed successfully. {stats['document_count']} documents indexed into {stats['node_count']} chunks."
            else:
                self.index_status.object = f"✅ Index created/refreshed successfully."
                
        except Exception as e:
            self.index_status.object = f"❌ Error creating index: {str(e)}"
            logger.error(f"Error creating index: {str(e)}")
            
        # Refresh folder lists
        self._refresh_folder_lists()
        
    def _load_selected_index(self, event):
        """Load the selected index."""
        folder_name = self.index_selector.value
        
        if not folder_name:
            self.query_result.object = "⚠️ Please select an index folder"
            return
            
        transcript_dir = os.path.join(self.base_transcript_dir, folder_name)
        storage_dir = os.path.join(self.base_storage_dir, folder_name, 'vector')
        
        self.query_result.object = f"⏳ Loading index from {folder_name}..."
        
        try:
            # Initialize LlamaIndexRAG with the selected folders
            self.llama_rag = LlamaIndexRAG(
                transcript_dir=transcript_dir,
                storage_dir=storage_dir
            )
            
            # Get stats
            stats = self.llama_rag.get_document_stats()
            if stats:
                self.query_result.object = f"✅ Index loaded successfully. {stats['document_count']} documents in {stats['node_count']} chunks. Ready to query!"
            else:
                self.query_result.object = f"✅ Index loaded successfully. Ready to query!"
                
        except Exception as e:
            self.query_result.object = f"❌ Error loading index: {str(e)}"
            logger.error(f"Error loading index: {str(e)}")
            
    def _handle_query(self, event):
        """Handle query submission."""
        if not self.llama_rag or not self.llama_rag.index:
            self.query_result.object = "⚠️ Please load an index first"
            return
            
        query_text = str(self.query_input.value or "").strip()
        
        if not query_text:
            self.query_result.object = "⚠️ Please enter a question"
            return
            
        if self.top_k_slider.value is None or not isinstance(self.top_k_slider.value, (int, float, str)):
            top_k = 2  # Default value
        else:
            top_k = int(self.top_k_slider.value)
        debug = bool(self.debug_checkbox.value)
        
        self.query_result.object = f"⏳ Processing query: '{query_text}'..."
        
        try:
            # Execute query
            response, source_nodes = self.llama_rag.query(query_text, similarity_top_k=top_k, debug=debug)
            
            # Display response
            self.query_result.object = f"### Response\n{response.response}"
            
            # Display sources
            if source_nodes:
                self.sources_title.visible = True
                
                source_panes = []
                for i, source in enumerate(source_nodes):
                    title = source.node.metadata.get('episode_title', f"Source {i+1}")
                    score = source.score
                    text = source.node.get_content()
                    
                    source_md = f"""
                    **Source {i+1}: {title}**  
                    *Relevance Score: {score:.4f}*
                    
                    ```
                    {text[:500]}...
                    ```
                    """
                    source_panes.append(pn.pane.Markdown(source_md, width=600))
                    
                self.sources_display.clear()
                self.sources_display.extend(source_panes)
            else:
                self.sources_title.visible = False
                self.sources_display.clear()
                
            # Display debug info if enabled
            if debug:
                debug_info = self.llama_rag.get_debug_info()
                if debug_info:
                    debug_md = """
                    ### Debug Information
                    
                    #### LLM Stats
                    ```
                    {llm_stats}
                    ```
                    
                    #### Embedding Stats
                    ```
                    {embed_stats}
                    ```
                    
                    #### Retrieval Stats
                    ```
                    {retrieval_stats}
                    ```
                    """.format(
                        llm_stats=str(debug_info["llm_events"]),
                        embed_stats=str(debug_info["embedding_events"]),
                        retrieval_stats=str(debug_info["retrieval_events"])
                    )
                    self.sources_display.append(pn.pane.Markdown(debug_md, width=600))
                    
        except Exception as e:
            self.query_result.object = f"❌ Error processing query: {str(e)}"
            logger.error(f"Error processing query: {str(e)}")
            self.sources_title.visible = False
            self.sources_display.clear()
            
    def get_app(self):
        """Get the Panel application."""
        return self.main_layout
        
def run_cli():
    """Run the Panel CLI application."""
    app = YouTubeRAGCLI()
    return app.get_app().servable()
    
if __name__ == "__main__":
    run_cli()