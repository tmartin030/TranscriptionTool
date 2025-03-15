import os
import logging
import argparse
import shutil
import torch
from datetime import datetime
import subprocess
from tqdm import tqdm
import sys

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.transcription.transcriber import Transcriber
from src.transcription.diarizer import Diarizer
from src.config.config import Config
from src.dataset.audio_dataset import AudioDataset
from src.document.document_generator import generate_transcript_document
from src.processing.nlp_postprocessing import NLPEngine
from torch.utils.data import DataLoader, Dataset

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

class SummaryDataset(Dataset):
    def __init__(self, transcript_items, nlp_engine):
        self.transcript_items = transcript_items
        self.nlp_engine = nlp_engine

    def __len__(self):
        return len(self.transcript_items)

    def __getitem__(self, idx):
        item = self.transcript_items[idx]
        all_transcript_text = " ".join([seg['cleaned_transcription'] if isinstance(seg['cleaned_transcription'], str) else str(seg['cleaned_transcription']) for seg in item['segments'] if seg['speaker'] != "TIMESTAMP"])
        return all_transcript_text, item

def main():
    parser = argparse.ArgumentParser(description="Transcription Tool")
    args = parser.parse_args()
    
    config_file = "src/config/config.json"
    config = Config(config_file)

    diarizer = Diarizer(config, config.get("diarization_model_dir"))
    transcriber = Transcriber(config, config.get("asr_model_dir"))
    nlp_engine = NLPEngine(config, config.get("nlp_model_dir"))

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
    audio_dataset = AudioDataset(audio_files_to_process_dicts, diarizer, transcriber, config, nlp_engine)
    
    audio_data_loader = DataLoader(
        audio_dataset,
        batch_size=1,   # Keep batch_size=1 per audio file to simplify handling
        shuffle=False,
    )

    # Data Structure for Document Generation
    transcript_items = []
    num_files = len(audio_data_loader)

    for i, (file_name_item, segments, transcriptions) in enumerate(audio_data_loader):
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
        last_timestamp = -30  # Initialize to -30 to ensure the first timestamp is printed.
        for j in range(len(segments)):
            start_time, end_time, speaker = segments[j]
            # Get the transcription dictionary for the current segment
            transcription_dict = transcriptions[j]
            # Extract the transcription text from the dictionary
            transcription = transcription_dict["transcription"]
            cleaned_transcription = transcription_dict["cleaned_transcription"]
            
            # Ensure speaker is a string before applying string methods
            if not isinstance(speaker, str):
                speaker = str(speaker)
                
            # Remove unwanted characters from speaker
            speaker = speaker.replace('(', '').replace(')', '').replace('\'', '')

            # Check if transcription is a list and handle it accordingly
            if isinstance(transcription, list):
                transcription = " ".join(transcription)  # Join list elements into a string
            elif not isinstance(transcription, str):
                transcription = str(transcription)

            # Remove brackets, quotes, and single quotes, but KEEP conjunction apostrophes
            transcription = transcription.replace('[', '').replace(']', '').replace('"', '')
            
            print(f"\rTranscribing file {i + 1} of {num_files} - Finished with segment {j + 1}/{len(segments)}", end="")
            segment_start_time = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{int(start_time % 60):02d}"

            # Timestamp logic
            if start_time - last_timestamp >= 30:
                timestamp_str = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{int(start_time % 60):02d}"
                segments_data.append({
                    'start_time': start_time,
                    'speaker': "TIMESTAMP",
                    'transcription': timestamp_str,
                    'cleaned_transcription': timestamp_str
                })
                last_timestamp = start_time

            # Remove comma after speaker and fix capitalization
            speaker = speaker.replace(",", "")
            if transcription.lower().startswith("just give us a few more minutes"):
                transcription = "Just give us a few more minutes."
            
            segments_data.append({
                'start_time': start_time,
                'speaker': speaker,
                'transcription': transcription,
                'cleaned_transcription': cleaned_transcription
            })
        
        transcript_items.append({
            'file_path': original_filename,
            'header_text': header_text,
            'transcription_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'segments': segments_data,
        })

        print(f"\nTranscription completed for {file_name} ({i + 1}/{num_files} files completed)")
    
    summary_dataset = SummaryDataset(transcript_items, nlp_engine)
    summary_data_loader = DataLoader(summary_dataset, batch_size=1)
    for all_transcript_text, item in tqdm(summary_data_loader, desc="Generating Summaries"):
        # Split the transcript into chunks
        max_length = 300
        print(f"Generating summary for {item['file_path']}")
        chunks = [all_transcript_text[i:i + max_length] for i in range(0, len(all_transcript_text), max_length)]
        summaries = []
        for chunk in chunks:
            summary = nlp_engine.generate_summary(chunk)
            summaries.append(summary)
        summary = " ".join(summaries)
        item['summary'] = summary

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
