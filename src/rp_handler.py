import os
import shutil
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup
from rp_schema import INPUT_VALIDATIONS
from predict import Predictor, Output
import os
import asyncio
import json
import urllib.parse
import subprocess
from typing import List, Dict, Any, Optional
import uuid

# Check if RUNPOD_ENDPOINT_ID exists in the environment
if os.getenv("RUNPOD_ENDPOINT_ID"):
    mode_to_run = "serverless"
else:
    # If RUNPOD_ENDPOINT_ID doesn't exist, use MODE_TO_RUN or default to "pod"
    mode_to_run = os.getenv("MODE_TO_RUN", "pod")
    
print(f"Starting in {mode_to_run} mode...")
    
MODEL = Predictor()
MODEL.setup()

def is_youtube_url(url: str) -> bool:
    """
    Check if the URL is a YouTube URL
    
    Args:
        url: URL to check
        
    Returns:
        bool: True if it's a YouTube URL
    """
    youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com']
    parsed_url = urllib.parse.urlparse(url)
    return any(domain == parsed_url.netloc for domain in youtube_domains)

def download_audio(url: str, output_dir: str = "downloads") -> List[str]:
    """
    Download audio from a YouTube URL (video or playlist)
    
    Args:
        url: YouTube URL (video or playlist)
        output_dir: Directory to save the downloaded files
    
    Returns:
        List[str]: List of paths to the downloaded files
    """
    downloaded_files = []
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine if it's a playlist URL
        is_playlist = "&list=" in url or "?list=" in url or "youtube.com/playlist" in url
        
        # We'll use the --print option to get the output filename
        # Create a temporary file format that includes metadata we need
        if is_playlist:
            output_template = f"{output_dir}/%(playlist)s/%(playlist_index)s-%(title)s.%(ext)s"
        else:
            output_template = f"{output_dir}/%(title)s.%(ext)s"
        
        # Base command
        command = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "0",  # 0 is the best quality
            "-o", output_template,
            # Print the final filename to stdout
            "--print", "after_move:filepath",
        ]
        
        if is_playlist:
            print("Detected playlist URL, downloading all videos in playlist...")
            command.append("--yes-playlist")  # Ensure it downloads the whole playlist
        
        # Add the URL
        command.append(url)
        
        # Run the command and capture output
        print(f"Downloading audio from: {url}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Parse the output to get file paths
        file_paths = result.stdout.strip().split('\n')
        downloaded_files = [path for path in file_paths if path and os.path.exists(path)]
        
        print(f"Download completed successfully! Downloaded {len(downloaded_files)} files.")
        return downloaded_files
        
    except subprocess.CalledProcessError as e:
        print(f"Error during download: {e}")
        if hasattr(e, 'stdout') and e.stdout:
            # Try to extract any successfully downloaded files from the output
            file_paths = e.stdout.strip().split('\n')
            downloaded_files = [path for path in file_paths if path and os.path.exists(path)]
            print(f"Partially completed, downloaded {len(downloaded_files)} files before error.")
        return downloaded_files
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return downloaded_files

def cleanup_job_files(job_id, jobs_directory='/jobs'):
    job_path = os.path.join(jobs_directory, job_id)
    if os.path.exists(job_path):
        try:
            shutil.rmtree(job_path)
            print(f"Removed job directory: {job_path}")
        except Exception as e:
            print(f"Error removing job directory {job_path}: {str(e)}")
    else:
        print(f"Job directory not found: {job_path}")

def cleanup_downloaded_files(file_paths: List[str]):
    """
    Clean up downloaded files and their parent directories if empty
    
    Args:
        file_paths: List of file paths to clean up
    """
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Removed file: {file_path}")
                
                # Try to remove parent directory if empty
                parent_dir = os.path.dirname(file_path)
                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
                    print(f"Removed empty directory: {parent_dir}")
            except Exception as e:
                print(f"Error removing file {file_path}: {str(e)}")

def run(job):
    # Ensure job has an id for testing in pod mode
    if 'id' not in job:
        job['id'] = str(uuid.uuid4())
        
    job_input = job['input']
    job_id = job['id']
    downloaded_files = []
    
    try:
        print(f"Processing job: {job_id}")
        
        # Input validation
        validated_input = validate(job_input, INPUT_VALIDATIONS)
        if 'errors' in validated_input:
            return {"error": validated_input['errors']}
        
        audio_file_paths = []
        
        # Check if we have a YouTube URL or a direct audio file URL
        if 'audio_file' in job_input:
            url = job_input['audio_file']
            
            if is_youtube_url(url):
                print(f"Detected YouTube URL: {url}")
                # Create a job-specific directory for downloads
                download_dir = os.path.join('/tmp', job_id)
                
                # Download audio from YouTube
                audio_file_paths = download_audio(url, download_dir)
                downloaded_files.extend(audio_file_paths)
                
                if not audio_file_paths:
                    return {"error": "Failed to download audio from YouTube URL"}
            else:
                # Regular audio file URL
                print(f"Downloading audio file from URL: {url}")
                audio_file_paths = [download_files_from_urls(job_id, [url])[0]]
        else:
            return {"error": "No audio_file URL provided"}
        
        # Initialize result container for multiple files
        results = []
        
        # Process each audio file
        for audio_file_path in audio_file_paths:
            print(f"Processing audio file: {audio_file_path}")
            
            # Prepare input for prediction
            predict_input = {
                'audio_file': audio_file_path,
                'language': job_input.get('language'),
                'language_detection_min_prob': job_input.get('language_detection_min_prob', 0),
                'language_detection_max_tries': job_input.get('language_detection_max_tries', 5),
                'initial_prompt': job_input.get('initial_prompt'),
                'batch_size': job_input.get('batch_size', 64),
                'temperature': job_input.get('temperature', 0),
                'vad_onset': job_input.get('vad_onset', 0.500),
                'vad_offset': job_input.get('vad_offset', 0.363),
                'align_output': job_input.get('align_output', False),
                'diarization': job_input.get('diarization', False),
                'huggingface_access_token': job_input.get('huggingface_access_token'),
                'min_speakers': job_input.get('min_speakers'),
                'max_speakers': job_input.get('max_speakers'),
                'debug': job_input.get('debug', False)
            }
            
            # Run prediction
            try:
                result = MODEL.predict(**predict_input)
                
                # Convert Output model to dict for JSON serialization
                output_dict = {
                    "file_path": os.path.basename(audio_file_path),
                    "segments": result.segments,
                    "detected_language": result.detected_language
                }
                
                results.append(output_dict)
                print(f"Successfully processed file: {audio_file_path}")
            except Exception as e:
                error_msg = f"Error processing file {audio_file_path}: {str(e)}"
                print(error_msg)
                results.append({
                    "file_path": os.path.basename(audio_file_path),
                    "error": error_msg
                })
        
        # Prepare the final response
        if len(results) == 1:
            # If only one file was processed, keep backward compatibility
            response = results[0]
        else:
            # For multiple files (e.g., playlist), return an array
            response = {
                "results": results,
                "total_files": len(results)
            }
        
        return response
        
    except Exception as e:
        error_msg = f"Error during job processing: {str(e)}"
        print(error_msg)
        return {"error": error_msg}
    
    finally:
        # Cleanup all downloaded files
        if downloaded_files:
            cleanup_downloaded_files(downloaded_files)
        
        # Cleanup other job files
        rp_cleanup.clean(['input_objects'])
        cleanup_job_files(job_id)
        print(f"Job {job_id} completed and cleanup finished")

# Start in the appropriate mode
if mode_to_run == "serverless":
    print("Starting in serverless mode...")
    runpod.serverless.start({"handler": run})
else:
    print("Starting in pod test mode...")
    
    async def test_main():
        # Create a test job
        test_job_id = str(22111)
        
        # Example audio URL from the sample GitHub repo
        test_audio_url = "https://www.youtube.com/watch?v=PtcQwb1uBhc"
        
        # Test with a regular audio file
        test_job = {
            "id": test_job_id,
            "input": {
                "audio_file": test_audio_url,
                "align_output": True

            }
        }
        
        print(f"Running test job: {test_job_id}")
        print(f"Test input: {json.dumps(test_job['input'], indent=2)}")
        
        # Run the job
        result = run(test_job)

    # Run the test
    asyncio.run(test_main())