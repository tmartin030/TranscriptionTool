from typing import List, Tuple
import os
import librosa  # Import librosa

import numpy as np
import torch
from torch.utils.data import Dataset

from src.transcription.transcriber import Transcriber
from src.transcription.diarizer import Diarizer
from src.config.config import Config

class AudioDataset(Dataset):
    def __init__(
        self,
        audio_files: List[str],  # Corrected: Now takes a list of file paths
        diarizer: Diarizer,
        transcriber: Transcriber,
        config: Config,
    ):
        self.audio_files = audio_files  # Corrected: Directly assign the list
        self.diarizer = diarizer
        self.transcriber = transcriber
        self.config = config
        self.audio_samples_per_segment = config.get(
            "audio_samples_per_segment", 30 * 16000
        )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[str, List, List]:
        audio_path = self.audio_files[idx]

        # Load audio using librosa
        audio_data, sampling_rate = librosa.load(audio_path, sr=None)

        # Pad or truncate the audio to match the model's expectation
        if len(audio_data) < self.audio_samples_per_segment:
            audio_data = np.pad(
                audio_data, (0, self.audio_samples_per_segment - len(audio_data))
            )
        else:
            audio_data = audio_data[: self.audio_samples_per_segment]

        if len(audio_data) == 0:
            print(f"Warning: Skipped processing of empty audio file: {audio_path}")
            return None, None, None

        # Diarization
        segments = self.diarizer.diarize(audio_data, sampling_rate)

        # Transcription
        transcriptions = []
        for start, end, speaker in segments:
            start = int(start * sampling_rate)
            end = int(end * sampling_rate)
            segment_audio = audio_data[start:end]
            transcription = self.transcriber.transcribe(segment_audio, sampling_rate)
            transcriptions.append(transcription)

        return os.path.basename(audio_path), segments, transcriptions
