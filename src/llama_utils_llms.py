"""
LlamaIndex functionality for indexing and querying YouTube transcripts.
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
    Response, # Ensure Response is imported
)
# ### MODIFIED SECTION START ###
# MODIFIED: Conditional imports for LLM and Embedding models
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
# ### MODIFIED SECTION END ###
from llama_index.core.node_parser import SentenceSplitter # Not explicitly used, but good to keep if planned
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType # For debug
# from llama_index.core.memory import Memory # Not used in current setup

# Load environment variables
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
    """
    def __init__(
        self,
        transcript_dir: Optional[str] = None,
        storage_dir: Optional[str] = None,
        # ### MODIFIED SECTION START ###
        llm_provider: str = "openai", # NEW: "openai" or "gemini"
        llm_model_name: Optional[str] = None, # MODIFIED: Generic model name
        embedding_model_name: Optional[str] = None, # MODIFIED: Generic model name
        # ### MODIFIED SECTION END ###
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        temperature: float = 0.1, # MODIFIED: Default temperature slightly higher for potentially more varied responses
        max_tokens: int = 1024, # For OpenAI, Gemini might have different limits or ways to specify
        similarity_top_k: int = 2
    ):
        """
        Initialize the LlamaIndexRAG object.
        
        Args:
            transcript_dir: Directory containing transcript files
            storage_dir: Directory for storing the index
            llm_provider: "openai" or "gemini"
            llm_model_name: Name of the LLM model to use (provider-specific)
            embedding_model_name: Name of the embedding model to use (provider-specific)
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between chunks
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM generation
            similarity_top_k: Number of similar nodes to retrieve for queries
        """
        self.transcript_dir = transcript_dir or os.getenv('TRANSCRIPT_DIR', 'data/transcripts')
        
        if not storage_dir:
            transcript_base = os.path.basename(os.path.normpath(self.transcript_dir))
            self.storage_dir = os.path.join('data/storage', transcript_base, 'vector')
        else:
            self.storage_dir = storage_dir
            
        os.makedirs(self.transcript_dir, exist_ok=True)
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # ### MODIFIED SECTION START ###
        self.llm_provider = llm_provider.lower()
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        # ### MODIFIED SECTION END ###
        
        self.chunk_size = int(os.getenv('CHUNK_SIZE', chunk_size))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', chunk_overlap))
        self.temperature = float(os.getenv('TEMPERATURE', temperature))
        self.max_tokens = int(os.getenv('MAX_TOKENS', max_tokens)) # Note: Gemini's max_output_tokens is part of generation_config
        self.similarity_top_k = similarity_top_k
        
        self.debug_handler = None
        self.callback_manager = None
        
        self._setup_llama_index()
        self.index = self._load_or_create_index()
        
    def _setup_llama_index(self):
        """Set up LlamaIndex settings with LLM and embedding model based on provider."""
        logger.info(f"Setting up LlamaIndex with LLM provider: {self.llm_provider}")

        llm: Optional[LLM] = None
        embed_model: Optional[BaseEmbedding] = None

        # ### MODIFIED SECTION START ###
        if self.llm_provider == "openai":
            from llama_index.llms.openai import OpenAI
            from llama_index.embeddings.openai import OpenAIEmbedding

            _llm_model = self.llm_model_name or os.getenv('OPENAI_LLM_MODEL', 'gpt-3.5-turbo') # Changed default from o4-mini
            _embedding_model = self.embedding_model_name or os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
            
            logger.info(f"Using OpenAI LLM: {_llm_model}, Embedding: {_embedding_model}")
            llm = OpenAI(
                temperature=self.temperature, 
                max_tokens=self.max_tokens, # This is for OpenAI
                model=_llm_model
            )
            embed_model = OpenAIEmbedding(model=_embedding_model)

        elif self.llm_provider == "gemini":
            try:
                from llama_index.llms.gemini import Gemini
                from llama_index.embeddings.gemini import GeminiEmbedding
            except ImportError:
                logger.error("Gemini packages not found. Please install with `pip install llama-index-llms-gemini llama-index-embeddings-gemini`")
                raise

            _llm_model = self.llm_model_name or os.getenv('GEMINI_LLM_MODEL', 'models/gemini-1.5-pro-latest') # or 'models/gemini-pro'
            _embedding_model = self.embedding_model_name or os.getenv('GEMINI_EMBEDDING_MODEL', 'models/text-embedding-004') # or 'models/embedding-001'
            
            # Check for GOOGLE_API_KEY
            if not os.getenv('GOOGLE_API_KEY'):
                logger.error("GOOGLE_API_KEY environment variable not set. Gemini models will not work.")
                # We might want to raise an error here or let it fail during instantiation
                # For now, let it proceed and LlamaIndex will likely raise an error.
            
            logger.info(f"Using Gemini LLM: {_llm_model}, Embedding: {_embedding_model}")
            llm = Gemini(
                model_name=_llm_model,
                temperature=self.temperature, # Gemini also has temperature
                # For Gemini, max_tokens is often part of generation_config
                # generation_config={"max_output_tokens": self.max_tokens} # Example
            )
            # GeminiEmbedding might have different task_type for retrieval, check docs if issues arise
            embed_model = GeminiEmbedding(model_name=_embedding_model)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}. Choose 'openai' or 'gemini'.")

        if llm is None or embed_model is None:
            raise RuntimeError(f"Failed to initialize LLM or embedding model for provider {self.llm_provider}")

        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
        # ### MODIFIED SECTION END ###
    
    def _load_or_create_index(self):
        """Load an existing index or create a new one if it doesn't exist."""
        # Index path should be unique per provider/model combination if embeddings are incompatible
        # For simplicity, current storage_dir doesn't reflect this, so ensure models are compatible
        # or use different storage_dirs if switching embeddings often.
        index_exists = os.path.exists(os.path.join(self.storage_dir, "docstore.json"))
        
        if index_exists:
            logger.info(f"Loading existing index from {self.storage_dir}")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
                return load_index_from_storage(storage_context)
            except Exception as e:
                logger.error(f"Error loading index: {str(e)}, attempting to create new index.")
                return self._create_new_index()
        else:
            logger.info(f"No existing index found at {self.storage_dir}, creating new index.")
            return self._create_new_index()
    
    def _create_new_index(self):
        """Create a new index from documents in the transcript directory."""
        if not os.path.exists(self.transcript_dir) or not os.listdir(self.transcript_dir):
            logger.warning(f"Transcript directory {self.transcript_dir} is empty or does not exist. Cannot create index.")
            return None
            
        logger.info(f"Creating new index from documents in {self.transcript_dir}")
        
        filename_fn = lambda filename: {'episode_title': os.path.splitext(os.path.basename(filename))[0]}
        
        documents = SimpleDirectoryReader(
            self.transcript_dir,
            filename_as_id=True,
            file_metadata=filename_fn
        ).load_data()
        
        for doc in documents:
            doc.excluded_llm_metadata_keys.append('episode_title')
            
        logger.info(f"Loaded {len(documents)} documents")
        
        if documents:
            try: # NEW: Added try-except for index creation
                index = VectorStoreIndex.from_documents(documents, show_progress=True)
                index.storage_context.persist(persist_dir=self.storage_dir)
                logger.info(f"Index created and persisted to {self.storage_dir}")
                return index
            except Exception as e: # NEW
                logger.error(f"Failed to create index: {e}", exc_info=True) # NEW
                return None # NEW
        else:
            logger.warning("No documents loaded, cannot create index.")
            return None
            
    def refresh_index(self):
        """Refresh the index with new or changed documents."""
        if not self.index:
            logger.warning("No index exists to refresh. Attempting to create a new one.")
            self.index = self._create_new_index()
            return # Return after attempting creation
            
        logger.info(f"Refreshing index with documents from {self.transcript_dir}")
        
        filename_fn = lambda filename: {'episode_title': os.path.splitext(os.path.basename(filename))[0]}
        
        current_documents = SimpleDirectoryReader(
            self.transcript_dir,
            filename_as_id=True,
            file_metadata=filename_fn
        ).load_data()
        
        for doc in current_documents:
            doc.excluded_llm_metadata_keys.append('episode_title')
            
        logger.info(f"Loaded {len(current_documents)} documents from disk for refresh")
        
        try: # NEW: Added try-except for refresh
            refreshed_results = self.index.refresh_ref_docs(
                current_documents,
                update_kwargs={"delete_kwargs": {'delete_from_docstore': True}}
            )
            
            changed_docs = sum(refreshed_results) if isinstance(refreshed_results, list) else 0 # Ensure it's a list
            if changed_docs > 0:
                logger.info(f"Updated {changed_docs} documents in the index")
                self.index.storage_context.persist(persist_dir=self.storage_dir)
                logger.info("Refreshed index persisted to disk")
            else:
                logger.info("No documents were added or changed during refresh.")
        except Exception as e: # NEW
            logger.error(f"Failed to refresh index: {e}", exc_info=True) # NEW

            
    def query(self, query_text: str, similarity_top_k: Optional[int] = None, debug: bool = False) -> Tuple[Response, List]:
        """
        Query the index with a natural language question.
        """
        if not self.index:
            logger.error("No index available for querying")
            # Ensure a LlamaIndex Response object is returned for consistency
            return Response(response="Error: No index loaded or available for querying.", source_nodes=[]), []
            
        if debug:
            self.debug_handler = LlamaDebugHandler(print_trace_on_end=True)
            self.callback_manager = CallbackManager([self.debug_handler])
            Settings.callback_manager = self.callback_manager # Apply to global settings
            logger.info("Debug handler enabled for this query")
        else: # NEW: Ensure callback manager is None if debug is false
            Settings.callback_manager = CallbackManager([]) # NEW: Use empty CallbackManager to satisfy type

        top_k_to_use = similarity_top_k if similarity_top_k is not None else self.similarity_top_k
        
        try: # NEW: Added try-except for query engine creation and query
            query_engine = self.index.as_query_engine(similarity_top_k=top_k_to_use)
            logger.info(f"Executing query: '{query_text}' with similarity_top_k={top_k_to_use} using {self.llm_provider}")
            response = query_engine.query(query_text)

            # Ensure response is of type Response (not AsyncStreamingResponse or other types)
            # This check might be overly cautious if your query_engine always returns Response
            if not isinstance(response, Response):
                logger.warning(f"Query returned type {type(response)}, attempting to coerce to Response object.")
                # Attempt to construct a Response object; this is a fallback
                response_text = str(response) # Get string representation
                # source_nodes might not be directly available on all response types
                source_nodes = getattr(response, "source_nodes", []) 
                response = Response(response=response_text, source_nodes=source_nodes)
            
            return response, getattr(response, "source_nodes", [])
        except Exception as e: # NEW
            logger.error(f"Error during query execution: {e}", exc_info=True) # NEW
            return Response(response=f"Error during query: {e}", source_nodes=[]), [] # NEW
        
    def get_debug_info(self):
        """Get debug info from the last query if debug was enabled."""
        if not self.debug_handler:
            return None
        # ... (rest of the method is fine)
        return {
            "llm_events": self.debug_handler.get_event_time_info(CBEventType.LLM),
            "embedding_events": self.debug_handler.get_event_time_info(CBEventType.EMBEDDING),
            "retrieval_events": self.debug_handler.get_event_time_info(CBEventType.RETRIEVE),
            "io_events": self.debug_handler.get_llm_inputs_outputs()
        }
        
    def count_documents(self):
        """Count the number of documents in the transcript directory."""
        if not os.path.exists(self.transcript_dir):
            return 0
        return len([f for f in os.listdir(self.transcript_dir) if f.endswith('.txt')])
        
    def get_document_stats(self):
        """Get statistics about the documents and chunks."""
        if not self.index or not hasattr(self.index, 'docstore'): # MODIFIED: Check for docstore
            return None
        try:
            node_count = len(self.index.docstore.docs)
            return {
                "document_count": self.count_documents(),
                "node_count": node_count
            }
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            return None