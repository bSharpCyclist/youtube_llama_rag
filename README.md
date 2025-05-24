# YouTube Llama RAG Project

## Overview

This project allows you to chat with YouTube videos using Large Language Models (LLMs) through a Streamlit web interface. It fetches YouTube video transcripts, processes them, and uses Retrieval Augmented Generation (RAG) to answer your questions about the video content. It supports LLMs from OpenAI and Google Gemini.

## Key Files

*   `youtube_utils.py`: Contains utility functions for fetching YouTube video transcripts.
*   `llama_utils_llms.py`: Manages the RAG pipeline, including data loading, indexing, and querying using LlamaIndex with different LLM providers.
*   `streamlit_app_llms.py`: The main Streamlit application file that provides the user interface for interacting with YouTube videos.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd youtube_llama_rag
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory by copying the `.env.example` file:
    ```bash
    cp .env.example .env
    ```
    Open the `.env` file and add your API keys:
    ```
    YOUTUBE_API_KEY="your_youtube_api_key_here" # Optional, used if transcript fetching via youtube-transcript-api fails for some videos
    OPENAI_API_KEY="your_openai_api_key_here"
    GEMINI_API_KEY="your_gemini_api_key_here"
    ```
    You can also update model names and other settings in this file if needed.

## Usage

Run the Streamlit application:

```bash
streamlit run streamlit_app_llms.py
```

Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).

Enter a YouTube video URL, select your desired LLM provider and model, and start asking questions!

## Configuration

The following environment variables can be configured in the `.env` file:

*   `YOUTUBE_API_KEY`: Your YouTube Data API v3 key (optional, for robust transcript fetching).
*   `OPENAI_API_KEY`: Your OpenAI API key.
*   `OPENAI_LLM_MODEL`: The OpenAI LLM model to use (e.g., `gpt-3.5-turbo`, `gpt-4`, `o4-mini`).
*   `OPENAI_EMBEDDING_MODEL`: The OpenAI embedding model to use (e.g., `text-embedding-ada-002`, `text-embedding-3-small`).
*   `GEMINI_API_KEY`: Your Google Gemini API key.
*   `GEMINI_LLM_MODEL`: The Gemini LLM model to use (e.g., `models/gemini-1.5-flash-latest`).
*   `GEMINI_EMBEDDING_MODEL`: The Gemini embedding model to use (e.g., `models/text-embedding-004`).
*   `CHUNK_SIZE`: Size of chunks for text splitting.
*   `CHUNK_OVERLAP`: Overlap between chunks.
*   `TEMPERATURE`: LLM temperature for generation.
*   `MAX_TOKENS`: Maximum tokens for LLM response.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.