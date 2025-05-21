# YouTube Transcript RAG Chatbot

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) system using LlamaIndex. It allows you to:

1. Download transcripts from YouTube videos or playlists
2. Create a searchable vector index from these transcripts
3. Query the indexed content through either:
   - A command-line interface (using Click)
   - A web-based chat interface (using Gradio)

## Setup

1. Clone the repository
2. Install the requirements:
```bash
pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and add your API keys:
```bash
cp .env.example .env
# Edit .env file with your actual API keys
```

## Required API Keys

- **OpenAI API Key**: Used for embeddings and completions
- **YouTube API Key**: Used to fetch playlist information (optional if you already have video IDs)

## Usage

### Command Line Interface (CLI)

```bash
# Run the CLI app
# Download YouTube transcripts
python -m src.click_app download --playlist PLY6oTPmKMsdZR6EjjYBBBlmYbyF4yYZwT --folder ancient-aliens
# OR use individual videos
python -m src.click_app download --videos "https://www.youtube.com/watch?v=dQw4w9WgXcQ" "https://www.youtube.com/watch?v=jNQXAC9IVRw" --folder my-videos

# Create an index from the downloaded transcripts
python -m src.click_app index --folder ancient-aliens

# Query the index
python -m src.click_app query --index ancient-aliens --question "What evidence supports ancient astronaut theory?" --top-k 3

# Show help information
python -m src.click_app --help
```

### Web Chat Interface

```bash
# Run the web interface
python -m src.ui_app
```

### All-in-one

```bash
# Run both interfaces
python -m src.main
```

## Features

- Download YouTube transcripts (individual videos or playlists)
- Create and manage vector indices for fast retrieval
- Query the index with natural language questions
- View source references from retrieved documents
- Interactive command-line interface with Panel
- Web-based chat interface with Gradio

## Project Structure

- `src/`: Python source code
  - `youtube_utils.py`: YouTube transcript downloading functionality
  - `llama_utils.py`: LlamaIndex setup and querying functionality
  - `cli_app.py`: Panel-based command-line interface
  - `ui_app.py`: Gradio-based web interface
  - `main.py`: Script to run both interfaces
- `data/`: Local data storage
  - `transcripts/`: Downloaded transcripts
  - `storage/`: Vector indices and related data