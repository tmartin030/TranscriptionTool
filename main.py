import os
import logging
import argparse
import shutil
import torch
import whisper  # Import the whisper library
from datetime import datetime
import subprocess
from tqdm import tqdm

from torch.utils.data import DataLoader

from src.transcription.diarizer import Diarizer
from src.config.config import Config
from src.dataset.audio_dataset import AudioDataset
from src.document.document_generator import generate_transcript_document

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load the Whisper model globally
model = whisper.load_model("large", device="cuda" if torch.cuda.is_available() else "cpu")

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

def transcribe_audio(audio_path):
    """
    Transcribes audio using the Whisper model.

    Args:
        audio_path: Path to the audio file.

    Returns:
        The transcription result from Whisper.
    """
    print("Transcribing audio...")
    result = model.transcribe(
        audio_path,
        language="en",  # Specify language explicitly, helps to improve transcription accuracy
        temperature=0.0, # Set temperature to 0.0 for best results. A value of 0.0 means the model will take the most likely prediction at each step, minimizing variability. Higher values introduce more creative or diverse results but may reduce accuracy.
        compression_ratio_threshold=2.1, # This helps handle text with high compression ratios (e.g., gibberish or highly repetitive text). If the generated text exceeds this ratio, it may be discarded to ensure quality. Lower this value if you're getting overly compressed outputs.
        logprob_threshold=-1.0, # Set the log probability threshold. A lower value will increase the number of words transcribed but may also increase the number of errors. A higher value will reduce the number of words transcribed but may also reduce the number of errors.
        no_speech_threshold=0.3 # Set the threshold for no speech detection. A higher value will reduce the number of false positives but may also reduce the
    )
    return result

def main():
    parser = argparse.ArgumentParser(description="Transcription Tool")
    args = parser.parse_args()
    
    config_file = "src/config/config.json"
    config = Config(config_file)

    diarizer = Diarizer(config, config.get("diarization_model_dir"))
    #transcriber = Transcriber(config, config.get("asr_model_dir")) # Removed custom transcriber

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
    audio_dataset = AudioDataset(audio_files_to_process_dicts, diarizer, config) # Removed transcriber
    
    data_loader = DataLoader(
        audio_dataset,
        batch_size=1,   # Keep batch_size=1 per audio file to simplify handling
        shuffle=False,
    )

    # Data Structure for Document Generation
    transcript_items = []
    num_files = len(data_loader)

    for i, (file_name_item, segments) in enumerate(data_loader): # Removed transcriptions and _
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
        # last_timestamp = -30  # Removed last timestamp
        
        # Transcribe the entire audio file
        transcription_result = transcribe_audio(file_name)
        full_transcription = transcription_result["text"]

        for j in range(len(segments)):
            start_time, end_time, speaker = segments[j]
            
            # Ensure speaker is a string before applying string methods
            if not isinstance(speaker, str):
                speaker = str(speaker)
                
            # Remove unwanted characters from speaker
            speaker = speaker.replace('(', '').replace(')', '').replace('\'', '')
            
            print(f"\rTranscribing file {i + 1} of {num_files} - Finished with segment {j + 1}/{len(segments)}", end="")
            segment_start_time = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{int(start_time % 60):02d}"

            # Timestamp logic Removed
            # if start_time - last_timestamp >= 30:
            #     timestamp_str = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{int(start_time % 60):02d}"
            #     segments_data.append({
            #         'start_time': start_time,
            #         'speaker': "TIMESTAMP",
            #         'transcription': timestamp_str
            #     })
            #     last_timestamp = start_time

            # Remove comma after speaker and fix capitalization
            speaker = speaker.replace(",", "")
            
            # Extract the segment's transcription from the full transcription
            segment_transcription = ""
            for segment in transcription_result["segments"]:
                if segment["start"] >= start_time and segment["end"] <= end_time:
                    segment_transcription += segment["text"] + " "
            
            if segment_transcription.lower().startswith("just give us a few more minutes"):
                segment_transcription = "Just give us a few more minutes."
            
            segments_data.append({
                'start_time': start_time,
                'speaker': f"Speaker {speaker}", # Added speaker
                'transcription': segment_transcription.strip()
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
