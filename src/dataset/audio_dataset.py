import os
import torch
import torchaudio
from torch.utils.data import Dataset
from src.config.config import Config
from src.transcription.diarizer import Diarizer
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, audio_files: list, diarizer: Diarizer, config: Config):
        self.audio_files = audio_files
        self.diarizer = diarizer
        self.config = config
        self.temp_dir = config.get("temp_dir")
        self.sampling_rate = 16000
        self.audio_cache = {}

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_name_item = self.audio_files[idx]
        if isinstance(file_name_item, dict):
            file_name = file_name_item.get("audio")
        elif isinstance(file_name_item, str):
            file_name = file_name_item
        else:
            raise ValueError(f"Invalid file name item: {file_name_item}")

        if file_name in self.audio_cache:
            audio_data = self.audio_cache[file_name]
        else:
            waveform, sample_rate = torchaudio.load(file_name)
            audio_data = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
            self.audio_cache[file_name] = audio_data

        waveform = audio_data["waveform"]
        sample_rate = audio_data["sample_rate"]

        # Resample if necessary
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sampling_rate)
            waveform = resampler(waveform)

        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Convert to numpy array
        audio_array = waveform.squeeze().numpy()

        # Diarize the audio
        segments = self.diarizer.diarize(file_name_item)

        return file_name_item, segments # Removed None
