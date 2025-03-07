import os

import torch
import torchaudio
from src.utils.gpu_utils import get_device
from src.processing.audio_loader import AudioLoader
from src.config.config import Config
from transformers import pipeline # changed import
import warnings

class Transcriber:
    def __init__(self, config: Config, model_dir):
        self.config = config
        self.device = get_device()
        self.model_name = config.get("asr_model")  # Get model name from config
        self.asr_model_dir = model_dir  # Only use this folder for storing models
        # removed this line
        # warnings.filterwarnings("ignore", message="Module 'speechbrain.pretrained' was deprecated")

        if not os.path.exists(self.asr_model_dir):
            os.makedirs(self.asr_model_dir)

        model_path = os.path.join(self.asr_model_dir, self.model_name.replace("/", "-"))
        # changed these lines
        print(f"Loading whisper model {self.model_name} to {self.asr_model_dir}...")
        self.asr_model = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=self.device
        )


    def transcribe(self, audio_file: str) -> str:
        """
        Performs transcription on the given audio file.

        Args:
            audio_file: Path to the audio file.

        Returns:
            The transcription as a string.
        """
        try:
            # removed this line
            #print("Using whisper model")
            # Transcribe using Whisper
            transcription = self.asr_model(audio_file)["text"].lower()
            return transcription
        except Exception as e:
            print(f"Error in transcription {audio_file}: {e}")
            return ""

    def transcribe_segment(self, audio_file: str, start: float, end: float) -> str:
        """
        Transcribes a specific segment of the audio file.

        Args:
            audio_file: Path to the audio file.
            start: Start time of the segment in seconds.
            end: End time of the segment in seconds.

        Returns:
            The transcription of the segment.
        """
        try:
            # removed this line
            #print("Using whisper model")
            # Transcribe using Whisper
            audio_segment = AudioLoader.load_audio_segment(audio_file, start, end)
            temp_audio_file = os.path.join(self.config.get("temp_dir"), "temp_audio_segment.wav")
            torchaudio.save(temp_audio_file, torch.from_numpy(audio_segment).unsqueeze(0).cpu(), 16000)
            transcription = self.asr_model(temp_audio_file)["text"].lower()
            return transcription
        except Exception as e:
            print(f"Error in segment transcription {audio_file}: {e}")
            return ""
