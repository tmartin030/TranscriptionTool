import os
import torch
import torchaudio
from transformers import pipeline
from src.utils.gpu_utils import get_device
from src.config.config import Config


class Transcriber:
    def __init__(self, config: Config, model_dir):
        self.config = config
        self.device = get_device()
        self.model_name = config.get("asr_model")
        self.model_dir = model_dir
        self.temp_dir = config.get("temp_dir")

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        model_path = os.path.join(self.model_dir, self.model_name.replace("/", "-"))

        if not os.path.exists(model_path):
            print(f"Downloading {self.model_name} to {self.model_dir}...")
            self.asr_model = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=self.device,
            )
            self.asr_model.save_pretrained(model_path)
        else:
            print(f"Loading existing {self.model_name} from {self.model_dir}...")
            self.asr_model = pipeline(
                "automatic-speech-recognition",
                model=model_path,
                device=self.device,
            )

    def transcribe(self, audio_segment: torch.Tensor) -> str:
        """
        Transcribes the given audio segment.

        Args:
            audio_segment: The audio segment as a torch.Tensor.

        Returns:
            The transcription as a string.
        """
        return self.transcribe_segment(audio_segment)

    def transcribe_segment(self, audio_segment: torch.Tensor) -> str:
        """
        Transcribes a segment of the given audio file.

        Args:
            audio_segment: The segment of the audio file as a tensor.

        Returns:
            The transcription as a string.
        """
        try:
            temp_audio_file = os.path.join(self.temp_dir, "temp_audio_segment.wav")
            torchaudio.save(temp_audio_file, audio_segment.unsqueeze(0).cpu(), 16000)
            transcription = self.asr_model(temp_audio_file, generate_kwargs={"task": "transcribe", "language": "english"})["text"].lower()
            return transcription
        except Exception as e:
            print(f"Error in segment transcription: {e}")
            return ""
