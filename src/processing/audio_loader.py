import os
from typing import Tuple

import librosa
import soundfile as sf
import numpy as np

class AudioLoader:
    @staticmethod
    def load_audio(audio_file: str) -> Tuple[np.ndarray, int]:
        """
        Loads an audio file.

        Args:
            audio_file: Path to the audio file.

        Returns:
            A tuple containing the audio signal (as a NumPy array) and the sample rate.
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        try:
            signal, rate = sf.read(audio_file)
        except Exception as e:
            raise RuntimeError(f"Error reading audio file {audio_file}: {e}")
        
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=0) #convert audio into mono if needed

        return signal, rate

    @staticmethod
    def load_audio_segment(audio_file: str, start: float, end: float) -> np.ndarray:
        """
        Loads a segment of an audio file.

        Args:
            audio_file: Path to the audio file.
            start: Start time of the segment in seconds.
            end: End time of the segment in seconds.

        Returns:
            A NumPy array containing the audio segment.
        """
        try:
            y, sr = librosa.load(audio_file, sr=None, mono=True)
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            return y[start_sample:end_sample]
        except Exception as e:
            raise RuntimeError(f"Error reading audio segment from {audio_file}: {e}")
