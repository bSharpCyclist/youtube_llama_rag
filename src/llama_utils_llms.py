# llama_utils.py
"""
LlamaIndex functionality for indexing and querying YouTube transcripts,
with support for multiple LLM providers (OpenAI, Gemini) and
provider-specific index storage.
"""

import os
import logging
import sys
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    load_index_from_storage,
    SimpleDirectoryReader,
    Settings,
    Response,
)
from llama_index.core.llms import LLM # Abstract LLM type
from llama_index.core.embeddings import BaseEmbedding # Abstract Embedding type
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class LlamaIndexRAG:
    """
    A class to handle LlamaIndex RAG operations for YouTube transcripts.
    Supports OpenAI and Gemini providers with provider-specific index storage.
    """
    def __init__(
        self,
        transcript_dir: str, # Made non-optional, should always be provided
        storage_dir_base: str = 'data/storage', # Base for all provider-specific storages
        llm_provider: str = "openai", 
        llm_model_name: Optional[str] = None, 
        embedding_model_name: Optional[str] = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        temperature: float = 0.1,
        max_tokens: int = 1024, # Note: For Gemini, this might be part of generation_config
        similarity_top_k: int = 2
    ):
        """
        Initialize the LlamaIndexRAG object.
        
        Args:
            transcript_dir: Directory containing transcript files (e.g., "data/transcripts/my_playlist").
            storage_dir_base: Base directory where provider-specific storages will be created 
                              (e.g., "data/storage").
            llm_provider: "openai" or "gemini".
            llm_model_name: Name of the LLM model to use (provider-specific).
            embedding_model_name: Name of the embedding model to use (provider-specific).
            chunk_size: Size of text chunks for indexing.
            chunk_overlap: Overlap between chunks.
            temperature: Temperature for LLM generation.
            max_tokens: Maximum tokens for LLM generation (OpenAI specific, Gemini uses generation_config).
            similarity_top_k: Number of similar nodes to retrieve for queries.
        """
        if not transcript_dir:
            raise ValueError("transcript_dir must be provided.")
            
        self.transcript_dir = os.path.normpath(transcript_dir)
        self.llm_provider = llm_provider.lower()
        
        # Extract the specific name of the transcript set (e.g., 'my_playlist')
        # This assumes transcript_dir is like 'data/transcripts/folder_name' or just 'folder_name'
        transcript_set_name = os.path.basename(self.transcript_dir)
        if not transcript_set_name: # Should not happen if transcript_dir is valid
            transcript_set_name = "default_transcripts" 
            logger.warning(f"Could not determine transcript set name from '{self.transcript_dir}', using '{transcript_set_name}'.")

        # Construct provider-specific storage path
        # e.g., data/storage/my_playlist/openai/vector
        self.storage_dir = os.path.join(
            storage_dir_base, 
            transcript_set_name,
            self.llm_provider,
            'vector'
        )
        logger.info(f"Using provider-specific storage directory: {self.storage_dir}")
            
        os.makedirs(self.transcript_dir, exist_ok=True)
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        
        self.chunk_size = int(os.getenv('CHUNK_SIZE', chunk_size))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', chunk_overlap))
        self.temperature = float(os.getenv('TEMPERATURE', temperature))
        self.max_tokens = int(os.getenv('MAX_TOKENS', max_tokens))
        self.similarity_top_k = similarity_top_k
        
        self.debug_handler = None
        self.callback_manager = None
        
        self._setup_llama_index()
        self.index = self._load_or_create_index()
        
    def _setup_llama_index(self):
        """Set up LlamaIndex global settings with LLM and embedding model based on provider."""
        logger.info(f"Setting up LlamaIndex with LLM provider: {self.llm_provider}")

        llm_instance: Optional[LLM] = None
        embed_model_instance: Optional[BaseEmbedding] = None

        if self.llm_provider == "openai":
            from llama_index.llms.openai import OpenAI
            from llama_index.embeddings.openai import OpenAIEmbedding

            # Prioritize model names passed to constructor, then env vars, then defaults
            _llm_model = self.llm_model_name or os.getenv('OPENAI_LLM_MODEL') or 'gpt-3.5-turbo'
            _embedding_model = self.embedding_model_name or os.getenv('OPENAI_EMBEDDING_MODEL') or 'text-embedding-3-small'
            
            logger.info(f"Using OpenAI LLM: {_llm_model}, Embedding: {_embedding_model}")
            if not os.getenv('OPENAI_API_KEY'):
                logger.warning("OPENAI_API_KEY environment variable not set. OpenAI models may fail.")

            llm_instance = OpenAI(
                temperature=self.temperature, 
                max_tokens=self.max_tokens,
                model=_llm_model
            )
            embed_model_instance = OpenAIEmbedding(model=_embedding_model)

        elif self.llm_provider == "gemini":
            try:
                from llama_index.llms.gemini import Gemini
                from llama_index.embeddings.gemini import GeminiEmbedding
            except ImportError:
                logger.error("Gemini packages not found. Please install with `pip install llama-index-llms-gemini llama-index-embeddings-gemini google-generativeai`")
                raise

            _llm_model = self.llm_model_name or os.getenv('GEMINI_LLM_MODEL') or 'models/gemini-1.5-flash-latest' # Using flash for speed/cost
            _embedding_model = self.embedding_model_name or os.getenv('GEMINI_EMBEDDING_MODEL') or 'models/text-embedding-004'
            
            logger.info(f"Using Gemini LLM: {_llm_model}, Embedding: {_embedding_model}")
            if not os.getenv('GOOGLE_API_KEY'):
                logger.warning("GOOGLE_API_KEY environment variable not set. Gemini models may fail.")
            
            llm_instance = Gemini(
                model_name=_llm_model,
                temperature=self.temperature,
                # generation_config={"max_output_tokens": self.max_tokens} # Example if needed
            )
            embed_model_instance = GeminiEmbedding(model_name=_embedding_model)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}. Choose 'openai' or 'gemini'.")

        if llm_instance is None or embed_model_instance is None:
            # This case should ideally be caught by the ValueError above or API key warnings
            raise RuntimeError(f"Failed to initialize LLM or embedding model for provider {self.llm_provider}")

        # Set LlamaIndex global settings
        Settings.llm = llm_instance
        Settings.embed_model = embed_model_instance
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
    
    def _load_or_create_index(self) -> Optional[VectorStoreIndex]:
        """Load an existing index from self.storage_dir or create a new one if it doesn't exist."""
        index_exists = os.path.exists(os.path.join(self.storage_dir, "docstore.json"))
        
        if index_exists:
            logger.info(f"Loading existing index from {self.storage_dir} for provider {self.llm_provider}")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
                index = load_index_from_storage(storage_context)
                logger.info(f"Successfully loaded index from {self.storage_dir}")
                return index if isinstance(index, VectorStoreIndex) else None
            except Exception as e:
                logger.error(f"Error loading index from {self.storage_dir}: {str(e)}. Attempting to create new index instead.", exc_info=True)
                return self._create_new_index()
        else:
            logger.info(f"No existing index found at {self.storage_dir} for provider {self.llm_provider}. Creating new index.")
            return self._create_new_index()
    
    def _create_new_index(self) -> Optional[VectorStoreIndex]:
        """Create a new index from documents in the transcript directory."""
        if not os.path.exists(self.transcript_dir) or not os.listdir(self.transcript_dir):
            logger.warning(f"Transcript directory {self.transcript_dir} is empty or does not exist. Cannot create index.")
            return None
            
        logger.info(f"Creating new index from documents in {self.transcript_dir} using {self.llm_provider} embeddings.")
        
        filename_fn = lambda filename: {'episode_title': os.path.splitext(os.path.basename(filename))[0]}
        
        try:
            documents = SimpleDirectoryReader(
                self.transcript_dir,
                filename_as_id=True,
                file_metadata=filename_fn
            ).load_data()
        except Exception as e:
            logger.error(f"Error reading documents from {self.transcript_dir}: {e}", exc_info=True)
            return None

        if not documents:
            logger.warning(f"No documents loaded from {self.transcript_dir}. Cannot create index.")
            return None

        for doc in documents:
            # Ensure 'episode_title' is excluded if it exists, handle potential absence
            if 'episode_title' not in doc.excluded_llm_metadata_keys:
                 doc.excluded_llm_metadata_keys.append('episode_title')
            
        logger.info(f"Loaded {len(documents)} documents for indexing.")
        
        try:
            index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True # Progress bar in console
            )
            index.storage_context.persist(persist_dir=self.storage_dir)
            logger.info(f"Index created and persisted to {self.storage_dir}")
            return index
        except Exception as e:
            logger.error(f"Failed to create or persist index: {e}", exc_info=True)
            return None
            
    def refresh_index(self):
        """Refresh the index with new or changed documents."""
        if not self.index:
            logger.warning("No index object exists to refresh. Attempting to create a new one.")
            self.index = self._create_new_index() # This will use the current provider settings
            return 
            
        logger.info(f"Refreshing index at {self.storage_dir} with documents from {self.transcript_dir} using {self.llm_provider} embeddings.")
        
        filename_fn = lambda filename: {'episode_title': os.path.splitext(os.path.basename(filename))[0]}
        
        try:
            current_documents = SimpleDirectoryReader(
                self.transcript_dir,
                filename_as_id=True,
                file_metadata=filename_fn
            ).load_data()
        except Exception as e:
            logger.error(f"Error reading documents from {self.transcript_dir} for refresh: {e}", exc_info=True)
            return

        if not current_documents:
            logger.warning(f"No documents found in {self.transcript_dir} for refresh. Index not changed.")
            return

        for doc in current_documents:
            if 'episode_title' not in doc.excluded_llm_metadata_keys:
                 doc.excluded_llm_metadata_keys.append('episode_title')
            
        logger.info(f"Loaded {len(current_documents)} documents from disk for refresh.")
        
        try:
            refreshed_results = self.index.refresh_ref_docs(
                current_documents,
                update_kwargs={"delete_kwargs": {'delete_from_docstore': True}} # Important for proper cleanup
            )
            
            # refreshed_results is a list of booleans indicating if a doc was updated/added/deleted
            changed_docs_count = sum(refreshed_results)
            
            if changed_docs_count > 0:
                logger.info(f"Index refresh affected {changed_docs_count} document states (added/removed/updated).")
                self.index.storage_context.persist(persist_dir=self.storage_dir)
                logger.info(f"Refreshed index persisted to {self.storage_dir}")
            else:
                logger.info("No documents were added, removed, or changed during refresh.")
        except Exception as e:
            logger.error(f"Failed to refresh index: {e}", exc_info=True)
            
    def query(self, query_text: str, similarity_top_k: Optional[int] = None, debug: bool = False) -> Tuple[Response, List[Any]]:
        """
        Query the index with a natural language question.
        
        Args:
            query_text: The question to ask.
            similarity_top_k: Number of similar nodes to retrieve (overrides instance default if provided).
            debug: Whether to enable LlamaIndex debug logging for this query.
            
        Returns:
            Tuple of (LlamaIndex Response object, list of source_nodes).
        """
        if not self.index:
            logger.error("No index available for querying.")
            return Response(response="Error: No index loaded or available for querying.", source_nodes=[]), []
            
        if debug:
            if not self.debug_handler: # Create handler if it doesn't exist
                self.debug_handler = LlamaDebugHandler(print_trace_on_end=True)
            if not self.callback_manager or self.debug_handler not in self.callback_manager.handlers:
                self.callback_manager = CallbackManager([self.debug_handler])
            Settings.callback_manager = self.callback_manager
            logger.info("Debug handler enabled for this query.")
        else:
            Settings.callback_manager = CallbackManager([]) # Ensure it's off if not debugging this query

        top_k_to_use = similarity_top_k if similarity_top_k is not None else self.similarity_top_k
        
        try:
            query_engine = self.index.as_query_engine(similarity_top_k=top_k_to_use)
            logger.info(f"Executing query: '{query_text}' with similarity_top_k={top_k_to_use} using {self.llm_provider} LLM.")
            response_obj = query_engine.query(query_text)

            if not isinstance(response_obj, Response):
                logger.warning(f"Query returned type {type(response_obj)}, attempting to coerce to Response object.")
                response_text_str = str(response_obj) 
                source_nodes_list = getattr(response_obj, "source_nodes", []) 
                response_obj = Response(response=response_text_str, source_nodes=source_nodes_list)
            
            return response_obj, getattr(response_obj, "source_nodes", [])
        except Exception as e:
            logger.error(f"Error during query execution: {e}", exc_info=True)
            return Response(response=f"Error during query: {str(e)}", source_nodes=[]), []
        
    def get_debug_info(self) -> Optional[Dict[str, Any]]:
        """Get debug info from the last query if debug was enabled."""
        if not self.debug_handler:
            logger.info("Debug handler was not active for the last query or has been cleared.")
            return None
            
        # It's good practice to clear the handler's events after retrieving them if they are per-query
        # For now, just return them.
        try:
            info = {
                "llm_events": self.debug_handler.get_event_time_info(CBEventType.LLM),
                "embedding_events": self.debug_handler.get_event_time_info(CBEventType.EMBEDDING),
                "retrieval_events": self.debug_handler.get_event_time_info(CBEventType.RETRIEVE),
                "io_events": self.debug_handler.get_llm_inputs_outputs()
            }
            # Optionally clear events: self.debug_handler.clear_event_infos()
            return info
        except Exception as e:
            logger.error(f"Error retrieving debug info: {e}", exc_info=True)
            return None
        
    def count_documents(self) -> int:
        """Count the number of .txt documents in the transcript directory."""
        if not os.path.exists(self.transcript_dir):
            return 0
        try:
            return len([f for f in os.listdir(self.transcript_dir) if f.endswith('.txt')])
        except Exception as e:
            logger.error(f"Error counting documents in {self.transcript_dir}: {e}", exc_info=True)
            return 0
        
    def get_document_stats(self) -> Optional[Dict[str, int]]:
        """Get statistics about the documents and nodes in the current index."""
        if not self.index or not hasattr(self.index, 'docstore'):
            logger.info("No index or docstore available to get stats from.")
            return None
            
        try:
            # Number of nodes (chunks) in the index's document store
            node_count = len(self.index.docstore.docs)
            
            # Number of unique source documents (based on .txt files in transcript_dir)
            source_document_count = self.count_documents()
            
            return {
                "document_count": source_document_count, # Number of original .txt files
                "node_count": node_count                 # Number of chunks/nodes in the index
            }
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}", exc_info=True)
            return None