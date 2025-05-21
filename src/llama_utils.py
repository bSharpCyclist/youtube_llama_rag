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
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType
from llama_index.core import Response
from llama_index.core.memory import Memory

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
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        temperature: float = 0,
        max_tokens: int = 1024,
        similarity_top_k: int = 2
    ):
        """
        Initialize the LlamaIndexRAG object.
        
        Args:
            transcript_dir: Directory containing transcript files
            storage_dir: Directory for storing the index
            llm_model: LLM model name (default from env or o4-mini)
            embedding_model: Embedding model name (default from env or text-embedding-3-small)
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between chunks
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM generation
            similarity_top_k: Number of similar nodes to retrieve for queries
        """
        self.transcript_dir = transcript_dir or os.getenv('TRANSCRIPT_DIR', 'data/transcripts')
        
        # If storage_dir is not provided, create one based on the transcript directory name
        if not storage_dir:
            transcript_base = os.path.basename(os.path.normpath(self.transcript_dir))
            self.storage_dir = os.path.join('data/storage', transcript_base, 'vector')
        else:
            self.storage_dir = storage_dir
            
        # Ensure directories exist
        os.makedirs(self.transcript_dir, exist_ok=True)
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Set LLM and embedding models
        self.llm_model = llm_model or os.getenv('LLM_MODEL', 'o4-mini')
        self.embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
        
        # Set chunking parameters
        self.chunk_size = int(os.getenv('CHUNK_SIZE', chunk_size))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', chunk_overlap))
        
        # Set LLM parameters
        self.temperature = float(os.getenv('TEMPERATURE', temperature))
        self.max_tokens = int(os.getenv('MAX_TOKENS', max_tokens))
        
        # Set query parameters
        self.similarity_top_k = similarity_top_k
        
        # Debug handler
        self.debug_handler = None
        self.callback_manager = None
        
        # Initialize LlamaIndex settings
        self._setup_llama_index()
        
        # Load or create index
        self.index = self._load_or_create_index()
        
    def _setup_llama_index(self):
        """Set up LlamaIndex settings with LLM and embedding model."""
        logger.info(f"Setting up LlamaIndex with LLM model: {self.llm_model}, Embedding model: {self.embedding_model}")
        
        # Initialize LLM and embedding model
        llm = OpenAI(
            temperature=self.temperature, 
            max_tokens=self.max_tokens,
            model=self.llm_model
        )
        embed_model = OpenAIEmbedding(model=self.embedding_model)
        
        # Set global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
    
    def _load_or_create_index(self):
        """Load an existing index or create a new one if it doesn't exist."""
        index_exists = os.path.exists(os.path.join(self.storage_dir, "docstore.json"))
        
        if index_exists:
            logger.info(f"Loading existing index from {self.storage_dir}")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
                return load_index_from_storage(storage_context)
            except Exception as e:
                logger.error(f"Error loading index: {str(e)}")
                logger.info("Creating new index instead")
                return self._create_new_index()
        else:
            logger.info("No existing index found")
            return self._create_new_index()
    
    def _create_new_index(self):
        """Create a new index from documents in the transcript directory."""
        if not os.path.exists(self.transcript_dir) or not os.listdir(self.transcript_dir):
            logger.warning(f"Transcript directory {self.transcript_dir} is empty or does not exist.")
            return None
            
        logger.info(f"Creating new index from documents in {self.transcript_dir}")
        
        # Add filename as metadata
        filename_fn = lambda filename: {
            'episode_title': os.path.splitext(os.path.basename(filename))[0]
        }
        
        # Load documents
        documents = SimpleDirectoryReader(
            self.transcript_dir,
            filename_as_id=True,
            file_metadata=filename_fn
        ).load_data()
        
        # Exclude metadata from LLM context to save tokens
        for doc in documents:
            doc.excluded_llm_metadata_keys.append('episode_title')
            
        logger.info(f"Loaded {len(documents)} documents")
        
        # Create index
        if documents:
            index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True
            )
            
            # Persist index
            index.storage_context.persist(persist_dir=self.storage_dir)
            logger.info(f"Index created and persisted to {self.storage_dir}")
            return index
        else:
            logger.warning("No documents loaded")
            return None
            
    def refresh_index(self):
        """Refresh the index with new or changed documents."""
        if not self.index:
            logger.warning("No index exists to refresh")
            self.index = self._create_new_index()
            return
            
        logger.info(f"Refreshing index with documents from {self.transcript_dir}")
        
        # Add filename as metadata
        filename_fn = lambda filename: {
            'episode_title': os.path.splitext(os.path.basename(filename))[0]
        }
        
        # Load current documents
        current_documents = SimpleDirectoryReader(
            self.transcript_dir,
            filename_as_id=True,
            file_metadata=filename_fn
        ).load_data()
        
        # Exclude metadata from LLM context
        for doc in current_documents:
            doc.excluded_llm_metadata_keys.append('episode_title')
            
        logger.info(f"Loaded {len(current_documents)} documents from disk")
        
        # Refresh index
        refreshed_results = self.index.refresh_ref_docs(
            current_documents,
            update_kwargs={"delete_kwargs": {'delete_from_docstore': True}}
        )
        
        # Log changes
        changed_docs = sum(refreshed_results)
        if changed_docs > 0:
            logger.info(f"Updated {changed_docs} documents in the index")
            
            # Persist updated index
            self.index.storage_context.persist(persist_dir=self.storage_dir)
            logger.info("Refreshed index persisted to disk")
        else:
            logger.info("No documents were added or changed")
            
    def query(self, query_text: str, similarity_top_k: int = 0, debug: bool = False) -> Tuple[Response, List]:
        """
        Query the index with a natural language question.
        
        Args:
            query_text: The question to ask
            similarity_top_k: Number of similar nodes to retrieve (overrides default)
            debug: Whether to enable debug logging
            
        Returns:
            Tuple of (response, source_nodes)
        """
        if not self.index:
            logger.error("No index available for querying")
            empty_response = Response(response="", source_nodes=[])
            return empty_response, []
            
        # Set up debug handler if requested
        if debug:
            self.debug_handler = LlamaDebugHandler(print_trace_on_end=True)
            self.callback_manager = CallbackManager([self.debug_handler])
            Settings.callback_manager = self.callback_manager
            logger.info("Debug handler enabled for this query")
            
        # Use provided similarity_top_k or instance default
        top_k = similarity_top_k or self.similarity_top_k
        
        # Create query engine
        query_engine = self.index.as_query_engine(similarity_top_k=top_k)
        
        # Execute query
        logger.info(f"Executing query: '{query_text}' with similarity_top_k={top_k}")
        response = query_engine.query(query_text)

        # Ensure response is of type Response (not AsyncStreamingResponse)
        if not isinstance(response, Response):
            # For StreamingResponse or AsyncStreamingResponse, convert to string for the answer
            response_text = str(response)
            # source_nodes may not be available, so return empty list
            empty_response = Response(response=response_text, source_nodes=[])
            return empty_response, []
        else:
            return response, getattr(response, "source_nodes", [])
        
    def get_debug_info(self):
        """Get debug info from the last query if debug was enabled."""
        if not self.debug_handler:
            return None
            
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
        if not self.index:
            return None
            
        # Count nodes in the index
        try:
            node_count = len(self.index.docstore.docs)
            return {
                "document_count": self.count_documents(),
                "node_count": node_count
            }
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            return None