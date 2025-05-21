import click
import os
from .youtube_utils import save_transcripts_from_playlist, save_transcripts_from_video_list
from .llama_utils import LlamaIndexRAG

@click.group()
def cli():
    """YouTube Transcript RAG CLI - Search YouTube video content with LLMs."""
    pass

@cli.command()
@click.option('--playlist', help='YouTube playlist ID')
@click.option('--videos', multiple=True, help='YouTube video URLs')
@click.option('--folder', required=True, help='Destination folder name')
def download(playlist, videos, folder):
    """Download YouTube transcripts from playlist or videos."""
    base_transcript_dir = "data/transcripts"
    output_dir = os.path.join(base_transcript_dir, folder)
    
    click.echo(f"Downloading transcripts to {output_dir}...")
    
    if playlist:
        success_count = save_transcripts_from_playlist(os.getenv('YOUTUBE_API_KEY', ''), playlist, output_dir)
        click.echo(f"Downloaded {success_count} transcripts from playlist")
    elif videos:
        success_count = save_transcripts_from_video_list(videos, output_dir)
        click.echo(f"Downloaded {success_count} transcripts from videos")
    else:
        click.echo("Error: Please provide either a playlist ID or video URLs")

@cli.command()
@click.option('--folder', required=True, help='Transcript folder to index')
def index(folder):
    """Create or refresh an index from transcripts."""
    base_transcript_dir = "data/transcripts"
    base_storage_dir = "data/storage"
    
    transcript_dir = os.path.join(base_transcript_dir, folder)
    storage_dir = os.path.join(base_storage_dir, folder, 'vector')
    
    click.echo(f"Creating/refreshing index for {folder}...")
    
    try:
        llama_rag = LlamaIndexRAG(transcript_dir=transcript_dir, storage_dir=storage_dir)
        llama_rag.refresh_index()
        stats = llama_rag.get_document_stats()
        if stats:
            click.echo(f"Index created with {stats['document_count']} documents in {stats['node_count']} chunks")
        else:
            click.echo("Index created successfully")
    except Exception as e:
        click.echo(f"Error creating index: {str(e)}")

@cli.command()
@click.option('--index', required=True, help='Index folder to query')
@click.option('--question', required=True, help='Question to ask')
@click.option('--top-k', default=2, help='Number of sources to retrieve')
@click.option('--debug', is_flag=True, help='Show debug information')
def query(index, question, top_k, debug):
    """Query an indexed transcript collection."""
    base_transcript_dir = "data/transcripts"
    base_storage_dir = "data/storage"
    
    transcript_dir = os.path.join(base_transcript_dir, index)
    storage_dir = os.path.join(base_storage_dir, index, 'vector')
    
    try:
        llama_rag = LlamaIndexRAG(transcript_dir=transcript_dir, storage_dir=storage_dir)
        click.echo(f"Querying: '{question}'...")
        response, source_nodes = llama_rag.query(question, similarity_top_k=top_k, debug=debug)
        
        click.echo("\n--- RESPONSE ---")
        click.echo(response.response)
        
        click.echo("\n--- SOURCES ---")
        for i, source in enumerate(source_nodes):
            title = source.node.metadata.get('episode_title', f"Source {i+1}")
            score = source.score
            text = source.node.get_content()
            click.echo(f"\nSource {i+1}: {title} (Score: {score:.4f})")
            click.echo(f"Text: {text[:300]}...")
            
        if debug:
            debug_info = llama_rag.get_debug_info()
            if debug_info:
                click.echo("\n--- DEBUG INFO ---")
                click.echo(f"LLM: {debug_info['llm_events']}")
                click.echo(f"Embeddings: {debug_info['embedding_events']}")
                click.echo(f"Retrieval: {debug_info['retrieval_events']}")
                
    except Exception as e:
        click.echo(f"Error querying index: {str(e)}")

if __name__ == "__main__":
    cli()