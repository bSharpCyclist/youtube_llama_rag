"""
YouTube transcript downloading functionality.
This module provides functions to download transcripts from YouTube videos and playlists.
"""

import os
import googleapiclient.discovery
from youtube_transcript_api._api import YouTubeTranscriptApi
from tqdm import tqdm
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_video_id(url):
    """Extract video ID from a YouTube URL."""
    # Handle various YouTube URL formats
    if not url:
        return None
        
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard YouTube URLs
        r'(?:embed\/|v\/|youtu.be\/)([0-9A-Za-z_-]{11})',  # Embedded and youtu.be URLs
        r'^([0-9A-Za-z_-]{11})$'  # Just the video ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def save_transcript_from_video(video_id, output_dir, title=None):
    """
    Download transcript for a single video and save it to a file.
    
    Args:
        video_id (str): YouTube video ID
        output_dir (str): Directory to save the transcript
        title (str, optional): Video title. If not provided, uses video ID.
        
    Returns:
        str: Path to the saved transcript file, or None if there was an error
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Use the provided title or the video ID if none
    safe_title = title or video_id
    # Remove any non-alphanumeric characters from the title
    safe_title = "".join([c for c in safe_title if c.isalnum() or c.isspace()]).strip()
    filename = os.path.join(output_dir, f"{safe_title}.txt")
    
    # Check if file already exists
    if os.path.exists(filename):
        logger.info(f"Transcript for '{safe_title}' already exists at {filename}")
        return filename
    
    try:
        # Get the transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Save the transcript
        with open(filename, "w", encoding="utf-8") as file:
            for entry in transcript_list:
                file.write(entry['text'] + ' ')
                
        logger.info(f"Transcript saved to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error fetching transcript for video ID {video_id}: {str(e)}")
        return None

def save_transcripts_from_playlist(api_key, playlist_id, output_dir):
    """
    Download transcripts for all videos in a YouTube playlist.
    
    Args:
        api_key (str): YouTube API key
        playlist_id (str): YouTube playlist ID
        output_dir (str): Directory to save the transcripts
        
    Returns:
        int: Number of successfully downloaded transcripts
    """
    if not api_key:
        logger.error("YouTube API Key is not set. Skipping transcript download.")
        return 0
        
    # Build the YouTube API client
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    
    # Get all the videos in the playlist
    videos = []
    next_page_token = None
    
    logger.info(f"Fetching videos from playlist {playlist_id}...")
    
    while True:
        request = youtube.playlistItems().list(
            part="contentDetails,snippet",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        try:
            response = request.execute()
            
            # Add each video to the list
            for item in response["items"]:
                video_id = item["contentDetails"]["videoId"]
                video_title = item["snippet"]["title"]
                video_date = item["snippet"]["publishedAt"]
                videos.append((video_id, video_title, video_date))
                
            # Check if there are more videos to fetch
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
                
        except Exception as e:
            logger.error(f"Error fetching playlist data: {str(e)}")
            return 0
    
    # Sort videos by date (newest first)
    videos.sort(key=lambda x: x[2], reverse=True)
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")
    
    # Download transcripts
    success_count = 0
    logger.info(f"Downloading transcripts for {len(videos)} videos...")
    
    for video_id, video_title, _ in tqdm(videos, desc="Downloading transcripts"):
        try:
            safe_title = "".join([c for c in video_title if c.isalnum() or c.isspace()]).strip()
            filename = os.path.join(output_dir, f"{safe_title}.txt")
            
            # Skip if file already exists
            if os.path.exists(filename):
                logger.info(f"Transcript for '{safe_title}' already exists. Assuming older videos are also present.")
                break
                
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            with open(filename, "w", encoding="utf-8") as file:
                for entry in transcript_list:
                    file.write(entry['text'] + ' ')
                    
            logger.info(f"Transcript saved to {filename}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error fetching transcript for video ID {video_id} ({video_title}): {str(e)}")
    
    logger.info(f"Successfully downloaded {success_count} transcripts to {output_dir}")
    return success_count

def save_transcripts_from_video_list(video_urls, output_dir):
    """
    Download transcripts for a list of YouTube video URLs.
    
    Args:
        video_urls (list): List of YouTube video URLs or IDs
        output_dir (str): Directory to save the transcripts
        
    Returns:
        int: Number of successfully downloaded transcripts
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")
    
    success_count = 0
    logger.info(f"Downloading transcripts for {len(video_urls)} videos...")
    
    for url in tqdm(video_urls, desc="Downloading transcripts"):
        video_id = extract_video_id(url)
        if not video_id:
            logger.error(f"Could not extract video ID from URL: {url}")
            continue
            
        if save_transcript_from_video(video_id, output_dir):
            success_count += 1
    
    logger.info(f"Successfully downloaded {success_count} transcripts to {output_dir}")
    return success_count