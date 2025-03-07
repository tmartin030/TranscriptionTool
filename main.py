import os
import json
import logging
import argparse
import shutil
import torch
from datetime import datetime
import subprocess
from tqdm import tqdm

from torch.utils.data import DataLoader

from src.transcription.transcriber import Transcriber
from src.transcription.diarizer import Diarizer
from src.config.config import Config
from src.dataset.audio_dataset import AudioDataset
from src.document.document_generator import generate_transcript_document

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_audio(video_path, output_path):
    """
    Extracts audio from a video file using FFmpeg.

    Args:
        video_path: Path to the input video file.
        output_path: Path to save the extracted audio file.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # FFmpeg command to extract audio to WAV format (PCM)
        command = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",  # Disable video
            "-acodec",
            "pcm_s16le",  # Use WAV (PCM) codec
            "-ar",
            "16000",  # Sample rate: 16kHz
            "-ac",
            "1",      # Set the channel to mono
            output_path,
            "-y",     # Overwrite the file if it exists
        ]

        # Execute the command
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully extracted audio to: {output_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from {video_path}:")
        print(f"  Return code: {e.returncode}")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please ensure it is installed and in your system's PATH.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Transcription Tool")
    args = parser.parse_args()
    
    config_file = "src/config/config.json"
    config = Config(config_file)

    diarizer = Diarizer(config, config.get("diarization_model_dir"))
    transcriber = Transcriber(config, config.get("asr_model_dir"))

    AV_input_dir = config.get("AV_input_dir")
    temp_dir = config.get("temp_dir")
    transcripts_dir = config.get("transcripts_dir")

    # Make paths absolute
    AV_input_dir = os.path.abspath(AV_input_dir)
    temp_dir = os.path.abspath(temp_dir)
    transcripts_dir = os.path.abspath(transcripts_dir)

    if not os.path.exists(transcripts_dir):
        os.makedirs(transcripts_dir)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # List to store audio files to process (both extracted and original audio)
    audio_files_to_process = []

    # Process files in the AV_input_dir
    for filename in os.listdir(AV_input_dir):
        file_path = os.path.join(AV_input_dir, filename)

        if filename.endswith((".mp4", ".m4a")):
            # Extract audio from video files
            audio_filename = os.path.splitext(filename)[0] + ".wav"
            temp_audio_path = os.path.join(temp_dir, audio_filename)
            try:
                extract_audio(file_path, temp_audio_path)
                audio_files_to_process.append(temp_audio_path)
            except Exception as e:
                print(f"Skipping video file '{filename}' due to error: {e}")
        elif filename.endswith((".aac", ".mp3", ".wav", ".ogg")):
            # Add audio files directly
            audio_files_to_process.append(file_path)
        else:
            print(f"Skipping unsupported file: {filename}")

    # Remove duplicates and check for existing files
    unique_audio_files = []
    for file_path in audio_files_to_process:
        if file_path not in unique_audio_files and os.path.exists(file_path):
            unique_audio_files.append(file_path)
        else:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
            else:
                print(f"Warning: Duplicate file: {file_path}")
    audio_files_to_process = unique_audio_files

    # Create the dataset using the temp audio files and audio_files_to_process
    # Convert audio file paths to dictionaries that pyannote.audio expects
    audio_files_to_process_dicts = [{"audio": file_path} for file_path in audio_files_to_process]
    audio_dataset = AudioDataset(audio_files_to_process_dicts, diarizer, transcriber, config)
    # Create the DataLoader
    data_loader = DataLoader(audio_dataset, batch_size=1, shuffle=False)

    # Data Structure for Document Generation
    transcript_items = []
    num_files = len(data_loader)

    for i, (file_name_item, segments, transcriptions) in enumerate(data_loader):
        # Unwrap file_name_item if it’s a list or tuple.
        if isinstance(file_name_item, (list, tuple)):
            file_name_item = file_name_item[0]

        # If it's a dict, extract the "audio" key; if it's already a string, use it directly.
        if isinstance(file_name_item, dict):
            file_name = file_name_item.get("audio")
        elif isinstance(file_name_item, str):
            file_name = file_name_item
        else:
            print(f"Skipping invalid file name: {file_name_item}")
            continue

        # If file_name is a list, extract its first element.
        if isinstance(file_name, list):
            file_name = file_name[0]

        # Verify that the file exists.
        if not os.path.exists(file_name):
            print(f"Warning: File not found: {file_name}")
            continue

        print(f"Processing {file_name} ({i + 1}/{num_files})")
        original_filename = os.path.splitext(os.path.basename(file_name))[0]
        header_text = f"Transcript of {original_filename}"

        # Collect segment data for the document generator.
        segments_data = []
        for j in range(len(segments)):
            start_time, end_time, speaker = segments[j]
            transcription = transcriptions[j]
            print(f"\rTranscribing file {i + 1} of {num_files} - Finished with segment {j + 1}/{len(segments)}", end="")
            segment_start_time = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{int(start_time % 60):02d}"

            # Debugging and Verification:
            print(f"  - Segment {j+1} - Speaker: {speaker}, Start Time: {segment_start_time}, Transcription: {transcription}")
            if not transcription:
                print(f"  - WARNING: Empty transcription for segment {j+1} of file {file_name}")

            segments_data.append({
                'start_time': start_time,
                'speaker': speaker,
                'transcription': transcription
            })
        
        transcript_items.append({
            'file_path': original_filename,
            'header_text': header_text,
            'transcription_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'segments': segments_data
        })

        print(f"\nTranscription completed for {file_name} ({i + 1}/{num_files} files completed)")

    try:
        # Generate the document using the new function.
        full_document = generate_transcript_document(transcript_items)
        output_file = os.path.join(transcripts_dir, "all_transcripts.docx")
        full_document.save(output_file)
        print(f"All transcriptions saved to {output_file}")
    except Exception as e:
        print(f"Error generating document: {e}")

    # Clean up temp audio files.
    print("cleaning temp audio folder")
    shutil.rmtree(temp_dir)
    print("cleaned temp audio folder")
    print("Finished")

if __name__ == "__main__":
    main()
