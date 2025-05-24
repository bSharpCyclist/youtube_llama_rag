# llama_utils.py (formerly llama_utils_llms.py)
"""
LlamaIndex functionality for indexing and querying YouTube transcripts.
Supports VectorStoreIndex for Q&A and SummaryIndex for summarization.
Uses a single LLM provider for both embeddings and generation.
"""

import os
import logging
import sys
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex, 
    SummaryIndex, 
    StorageContext, 
    load_index_from_storage,
    SimpleDirectoryReader,
    Settings,
    Response,
    Document, 
)
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class LlamaIndexRAG:
    def __init__(
        self,
        transcript_dir: str,
        storage_dir_base: str = 'data/storage',
        llm_provider: str = "openai", 
        llm_model_name: Optional[str] = None, 
        embedding_model_name: Optional[str] = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        similarity_top_k: int = 3
    ):
        if not transcript_dir:
            raise ValueError("transcript_dir must be provided.")
            
        self.transcript_dir = os.path.normpath(transcript_dir)
        self.llm_provider = llm_provider.lower()
        
        transcript_set_name = os.path.basename(self.transcript_dir)
        if not transcript_set_name:
            transcript_set_name = "default_transcripts" 
            logger.warning(f"Could not determine transcript set name from '{self.transcript_dir}', using '{transcript_set_name}'.")

        self.provider_storage_base = os.path.join(
            storage_dir_base, 
            transcript_set_name,
            self.llm_provider
        )
        
        self.vector_storage_dir = os.path.join(self.provider_storage_base, 'vector_index')
        self.summary_storage_dir = os.path.join(self.provider_storage_base, 'summary_index')
        
        logger.info(f"Vector index storage directory: {self.vector_storage_dir}")
        logger.info(f"Summary index storage directory: {self.summary_storage_dir}")
            
        os.makedirs(self.transcript_dir, exist_ok=True)
        os.makedirs(self.vector_storage_dir, exist_ok=True)
        os.makedirs(self.summary_storage_dir, exist_ok=True)
        
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        
        self.chunk_size = int(os.getenv('CHUNK_SIZE', chunk_size))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', chunk_overlap))
        self.temperature = float(os.getenv('TEMPERATURE', temperature))
        self.max_tokens = int(os.getenv('MAX_TOKENS', max_tokens))
        self.similarity_top_k = similarity_top_k
        
        self.debug_handler = None
        self.callback_manager = None

        self.vector_index: Optional[VectorStoreIndex] = None
        self.summary_index: Optional[SummaryIndex] = None
        self.documents: Optional[List[Document]] = None
        
        self._setup_llama_index()
        self._load_or_create_indexes()
        
        # Keep self.index pointing to vector_index for any existing generic references
        # but prefer specific query methods (query_vector_index, query_summary_index)
        self.index = self.vector_index 
        
    def _setup_llama_index(self):
        logger.info(f"Setting up LlamaIndex with LLM provider: {self.llm_provider}")
        llm_instance: Optional[LLM] = None
        embed_model_instance: Optional[BaseEmbedding] = None

        if self.llm_provider == "openai":
            from llama_index.llms.openai import OpenAI
            from llama_index.embeddings.openai import OpenAIEmbedding
            _llm_model = self.llm_model_name or os.getenv('OPENAI_LLM_MODEL') or 'gpt-3.5-turbo'
            _embedding_model = self.embedding_model_name or os.getenv('OPENAI_EMBEDDING_MODEL') or 'text-embedding-3-small'
            logger.info(f"Using OpenAI LLM: {_llm_model}, Embedding: {_embedding_model}")
            if not os.getenv('OPENAI_API_KEY'): logger.warning("OPENAI_API_KEY not set.")
            llm_instance = OpenAI(temperature=self.temperature, max_tokens=self.max_tokens, model=_llm_model)
            embed_model_instance = OpenAIEmbedding(model=_embedding_model)
        elif self.llm_provider == "gemini":
            try:
                from llama_index.llms.gemini import Gemini
                from llama_index.embeddings.gemini import GeminiEmbedding
            except ImportError: 
                logger.error("Gemini packages not found. `pip install llama-index-llms-gemini llama-index-embeddings-gemini`")
                raise
            _llm_model = self.llm_model_name or os.getenv('GEMINI_LLM_MODEL') or 'models/gemini-1.5-flash-latest'
            _embedding_model = self.embedding_model_name or os.getenv('GEMINI_EMBEDDING_MODEL') or 'models/text-embedding-004'
            logger.info(f"Using Gemini LLM: {_llm_model}, Embedding: {_embedding_model}")
            if not os.getenv('GEMINI_API_KEY'): logger.warning("GEMINI_API_KEY not set.")
            llm_instance = Gemini(model_name=_llm_model, temperature=self.temperature)
            embed_model_instance = GeminiEmbedding(model_name=_embedding_model, task_type="RETRIEVAL_DOCUMENT")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}.")

        if llm_instance is None or embed_model_instance is None:
            raise RuntimeError(f"Failed to initialize LLM or embedding model for provider {self.llm_provider}")
        
        self.actual_llm_model_name = _llm_model
        self.actual_embedding_model_name = _embedding_model
        
        Settings.llm = llm_instance
        Settings.embed_model = embed_model_instance
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap

    def _load_documents(self) -> Optional[List[Document]]:
        if self.documents:
            return self.documents

        if not os.path.exists(self.transcript_dir) or not os.listdir(self.transcript_dir):
            logger.warning(f"Transcript directory {self.transcript_dir} is empty or does not exist.")
            return None
            
        logger.info(f"Loading documents from {self.transcript_dir}")
        filename_fn = lambda filename: {'episode_title': os.path.splitext(os.path.basename(filename))[0]}
        try:
            loaded_docs = SimpleDirectoryReader(
                self.transcript_dir,
                filename_as_id=True,
                file_metadata=filename_fn
            ).load_data()
            if not loaded_docs:
                logger.warning(f"No documents loaded from {self.transcript_dir}.")
                return None
            for doc in loaded_docs:
                if 'episode_title' not in doc.excluded_llm_metadata_keys:
                     doc.excluded_llm_metadata_keys.append('episode_title')
            self.documents = loaded_docs
            logger.info(f"Loaded {len(self.documents)} documents.")
            return self.documents
        except Exception as e:
            logger.error(f"Error reading documents from {self.transcript_dir}: {e}", exc_info=True)
            return None

    def _load_or_create_indexes(self):
        docs = self._load_documents()

        # VectorStoreIndex
        vector_index_exists = os.path.exists(os.path.join(self.vector_storage_dir, "docstore.json"))
        if vector_index_exists:
            logger.info(f"Loading existing VectorStoreIndex from {self.vector_storage_dir}")
            try:
                storage_context_vector = StorageContext.from_defaults(persist_dir=self.vector_storage_dir)
                loaded_index = load_index_from_storage(storage_context_vector)
                if isinstance(loaded_index, VectorStoreIndex):
                    self.vector_index = loaded_index
                    logger.info(f"Successfully loaded VectorStoreIndex.")
                else:
                    logger.warning(f"Loaded object from {self.vector_storage_dir} is not a VectorStoreIndex. Type: {type(loaded_index)}. Will attempt to create new.")
                    if docs: self.vector_index = self._create_vector_index(docs)
            except Exception as e:
                logger.error(f"Error loading VectorStoreIndex: {str(e)}. Attempting to create new.", exc_info=True)
                if docs: self.vector_index = self._create_vector_index(docs)
        else:
            logger.info(f"No existing VectorStoreIndex found at {self.vector_storage_dir}. Creating new one.")
            if docs: self.vector_index = self._create_vector_index(docs)

        # SummaryIndex
        summary_index_exists = os.path.exists(os.path.join(self.summary_storage_dir, "docstore.json"))
        if summary_index_exists:
            logger.info(f"Loading existing SummaryIndex from {self.summary_storage_dir}")
            try:
                storage_context_summary = StorageContext.from_defaults(persist_dir=self.summary_storage_dir)
                loaded_index = load_index_from_storage(storage_context_summary)
                if isinstance(loaded_index, SummaryIndex):
                    self.summary_index = loaded_index
                    logger.info(f"Successfully loaded SummaryIndex.")
                else:
                    logger.warning(f"Loaded object from {self.summary_storage_dir} is not a SummaryIndex. Type: {type(loaded_index)}. Will attempt to create new.")
                    if docs: self.summary_index = self._create_summary_index(docs)
            except Exception as e:
                logger.error(f"Error loading SummaryIndex: {str(e)}. Attempting to create new.", exc_info=True)
                if docs: self.summary_index = self._create_summary_index(docs)
        else:
            logger.info(f"No existing SummaryIndex found at {self.summary_storage_dir}. Creating new one.")
            if docs: self.summary_index = self._create_summary_index(docs)
        
        self.index = self.vector_index


    def _create_vector_index(self, documents: List[Document]) -> Optional[VectorStoreIndex]:
        logger.info(f"Creating new VectorStoreIndex using {self.llm_provider} embeddings.")
        try:
            index = VectorStoreIndex.from_documents(documents, show_progress=True)
            index.storage_context.persist(persist_dir=self.vector_storage_dir)
            logger.info(f"VectorStoreIndex created and persisted to {self.vector_storage_dir}")
            return index
        except Exception as e:
            logger.error(f"Failed to create or persist VectorStoreIndex: {e}", exc_info=True)
            return None

    def _create_summary_index(self, documents: List[Document]) -> Optional[SummaryIndex]:
        logger.info(f"Creating new SummaryIndex using {self.llm_provider} LLM.")
        try:
            index = SummaryIndex.from_documents(documents, show_progress=True)
            index.storage_context.persist(persist_dir=self.summary_storage_dir)
            logger.info(f"SummaryIndex created and persisted to {self.summary_storage_dir}")
            return index
        except Exception as e:
            logger.error(f"Failed to create or persist SummaryIndex: {e}", exc_info=True)
            return None
            
    def refresh_indexes(self):
        logger.info(f"Attempting to refresh indexes.")
        
        # Reload documents to ensure we have the latest for refresh/rebuild
        self.documents = None # Clear cached documents
        current_docs = self._load_documents()

        if not current_docs:
            logger.warning("No documents loaded, cannot refresh or rebuild indexes.")
            return

        # Refresh VectorStoreIndex
        if not self.vector_index:
            logger.warning("No VectorStoreIndex to refresh. Creating a new one.")
            self.vector_index = self._create_vector_index(current_docs)
        else:
            logger.info(f"Refreshing VectorStoreIndex at {self.vector_storage_dir}")
            try:
                # SimpleDirectoryReader re-reads for refresh_ref_docs to detect changes
                # We pass the already loaded current_docs which should be up-to-date
                refreshed_results = self.vector_index.refresh_ref_docs(
                    current_docs, # Pass the freshly loaded documents
                    update_kwargs={"delete_kwargs": {'delete_from_docstore': True}}
                )
                changed_docs_count = sum(refreshed_results)
                if changed_docs_count > 0:
                    logger.info(f"VectorStoreIndex refresh affected {changed_docs_count} document states.")
                    self.vector_index.storage_context.persist(persist_dir=self.vector_storage_dir)
                    logger.info(f"Refreshed VectorStoreIndex persisted.")
                else:
                    logger.info("No changes to VectorStoreIndex during refresh.")
            except Exception as e:
                logger.error(f"Failed to refresh VectorStoreIndex: {e}", exc_info=True)

        # Rebuild SummaryIndex
        logger.info(f"Rebuilding SummaryIndex at {self.summary_storage_dir} to ensure it's up-to-date.")
        self.summary_index = self._create_summary_index(current_docs) # Rebuild with fresh docs
        
        self.index = self.vector_index


    def query_vector_index(self, query_text: str, similarity_top_k: Optional[int] = None, debug: bool = False) -> Tuple[Response, List[Any]]:
        if not self.vector_index:
            logger.error("VectorStoreIndex not available for querying.")
            return Response(response="Error: Q&A index not loaded.", source_nodes=[]), []
            
        if debug:
            if not self.debug_handler: self.debug_handler = LlamaDebugHandler(print_trace_on_end=True)
            if not self.callback_manager or self.debug_handler not in self.callback_manager.handlers:
                self.callback_manager = CallbackManager([self.debug_handler])
            Settings.callback_manager = self.callback_manager
            logger.info("Debug handler enabled for this Q&A query.")
        else:
            Settings.callback_manager = CallbackManager([])

        top_k_to_use = similarity_top_k if similarity_top_k is not None else self.similarity_top_k
        
        try:
            query_engine = self.vector_index.as_query_engine(similarity_top_k=top_k_to_use)
            logger.info(f"Executing Q&A query: '{query_text}' with similarity_top_k={top_k_to_use} using {self.llm_provider} LLM.")
            response_obj = query_engine.query(query_text)

            if not isinstance(response_obj, Response):
                logger.warning(f"Q&A Query returned type {type(response_obj)}, coercing.")
                response_text_str = str(response_obj); source_nodes_list = getattr(response_obj, "source_nodes", []) 
                response_obj = Response(response=response_text_str, source_nodes=source_nodes_list)
            return response_obj, getattr(response_obj, "source_nodes", [])
        except Exception as e:
            logger.error(f"Error during Q&A query: {e}", exc_info=True)
            return Response(response=f"Error during Q&A query: {str(e)}", source_nodes=[]), []

    def query_summary_index(self, 
                            query_text: Optional[str] = None, 
                            output_format: str = "summary", # NEW: "summary" or "outline"
                            debug: bool = False) -> Tuple[Response, List[Any]]:
        if not self.summary_index:
            logger.error("SummaryIndex not available for querying.")
            return Response(response="Error: Summary index not loaded.", source_nodes=[]), []

        if debug:
            summary_debug_handler = LlamaDebugHandler(print_trace_on_end=True)
            Settings.callback_manager = CallbackManager([summary_debug_handler])
            logger.info("Debug handler enabled for this summary query.")
        else:
            Settings.callback_manager = CallbackManager([])

        # Determine the prompt based on desired output_format
        if output_format == "outline":
            # Prompt for a structured outline. You might need to experiment with this prompt.
            # Asking for Markdown can help with formatting.
            actual_query = query_text or \
                ("List the main topics of this content.")
        else: # Default to "summary"
            actual_query = query_text or "Provide a concise summary of all the content."
        
        logger.info(f"Executing summary_index query for '{output_format}': '{actual_query}' using {self.llm_provider} LLM.")
        
        try:
            # For SummaryIndex, response_mode="tree_summarize" is good for hierarchical tasks.
            # Other modes like "refine" or "compact" might also be experimented with.
            query_engine = self.summary_index.as_query_engine(
                response_mode="tree_summarize", 
                use_async=False,
                # verbose=True # Can be helpful for debugging summary_index behavior
            )
            response_obj = query_engine.query(actual_query)

            if not isinstance(response_obj, Response):
                logger.warning(f"Summary Query returned type {type(response_obj)}, coercing.")
                response_text_str = str(response_obj); source_nodes_list = getattr(response_obj, "source_nodes", [])
                response_obj = Response(response=response_text_str, source_nodes=source_nodes_list)
            
            # The response_obj.response will contain the LLM's attempt at the outline or summary.
            # source_nodes might not be as directly relevant for a generated outline as for Q&A.
            return response_obj, getattr(response_obj, "source_nodes", [])
        except Exception as e:
            logger.error(f"Error during summary_index query for {output_format}: {e}", exc_info=True)
            return Response(response=f"Error during summary_index query for {output_format}: {str(e)}", source_nodes=[]), []
        
    def get_debug_info(self) -> Optional[Dict[str, Any]]:
        # This will return info from the last debug handler that was active
        # (either from Q&A or Summary if they used self.debug_handler)
        # If summary query used its own local debug_handler, this won't capture it.
        # For simplicity, we assume self.debug_handler was used if debug=True for Q&A.
        if not self.debug_handler: 
            logger.info("Main debug handler was not active for the last Q&A query or has been cleared.")
            return None
        try:
            info = {
                "llm_events": self.debug_handler.get_event_time_info(CBEventType.LLM),
                "embedding_events": self.debug_handler.get_event_time_info(CBEventType.EMBEDDING),
                "retrieval_events": self.debug_handler.get_event_time_info(CBEventType.RETRIEVE),
                "io_events": self.debug_handler.get_llm_inputs_outputs()
            }
            return info
        except Exception as e:
            logger.error(f"Error retrieving debug info: {e}", exc_info=True)
            return None
        
    def count_documents(self) -> int:
        if not os.path.exists(self.transcript_dir): return 0
        try: return len([f for f in os.listdir(self.transcript_dir) if f.endswith('.txt')])
        except Exception as e:
            logger.error(f"Error counting documents in {self.transcript_dir}: {e}", exc_info=True)
            return 0
        
    def get_document_stats(self) -> Optional[Dict[str, Union[int, str]]]:
        doc_count = self.count_documents()
        node_count_str = "N/A"

        if self.vector_index and hasattr(self.vector_index, 'docstore'):
            try:
                node_count_str = str(len(self.vector_index.docstore.docs))
            except Exception as e:
                logger.error(f"Error getting node count from VectorStoreIndex: {e}")
        elif self.summary_index: # If no vector index, at least show doc count
             pass # node_count_str remains "N/A (SummaryIndex)" or similar

        if doc_count > 0 or node_count_str != "N/A":
            return {"document_count": doc_count, "node_count": node_count_str}
        
        logger.info("No index available to get stats from.")
        return None