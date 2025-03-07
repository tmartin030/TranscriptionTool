import os
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset  # Import PyTorch's Dataset class

class AudioDataset(Dataset):
    """
    A PyTorch dataset for processing audio files. It loads the audio, pads/truncates it, converts to mono,
    and then performs diarization and transcription.
    """
    def __init__(self, audio_files, diarizer, transcriber, config):
        self.audio_files = audio_files
        self.diarizer = diarizer
        self.transcriber = transcriber
        self.config = config
        self.audio_samples_per_segment = config.get("audio_samples_per_segment", 30 * 16000)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx: int):
        audio_item = self.audio_files[idx]
        # If the audio_item is a dict, extract the file path from the "audio" key
        if isinstance(audio_item, dict):
            audio_path = audio_item.get("audio")
        else:
            audio_path = audio_item

        # Ensure audio_path is a string
        if not isinstance(audio_path, str):
            print(f"Warning: Audio path is not a string: {audio_path}")
            return audio_path, [], []

        try:
            audio_data, sampling_rate = sf.read(audio_path)
        except Exception as e:
            print(f"Warning: Error reading audio file: {audio_path} - {e}")
            return audio_path, [], []

        # Handle empty audio files
        if len(audio_data) == 0:
            print(f"Warning: Skipped processing of empty audio file: {audio_path}")
            return audio_path, [], []

        # Pad or truncate the audio
        if len(audio_data) < self.audio_samples_per_segment:
            audio_data = np.pad(audio_data, (0, self.audio_samples_per_segment - len(audio_data)))
        else:
            audio_data = audio_data[: self.audio_samples_per_segment]

        # Change to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Convert the numpy array to a torch tensor and add a channel dimension (1, time)
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)

        # Create the input dictionary for diarization
        diarization_input = {"waveform": audio_tensor, "sample_rate": sampling_rate}

        # Diarization: now pass the dictionary to the diarizer
        segments = self.diarizer.diarize(diarization_input)

        # Transcription: process each segment on the original audio_data
        transcriptions = []
        for start, end, speaker in segments:
            start_idx = int(start * sampling_rate)
            end_idx = int(end * sampling_rate)
            segment_audio = audio_data[start_idx:end_idx]
            transcription = self.transcriber.transcribe(segment_audio)
            transcriptions.append(transcription)
        
        return audio_path, segments, transcriptions
