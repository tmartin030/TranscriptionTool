import os
import torch
import torchaudio
from transformers import pipeline, AutoTokenizer
from src.utils.gpu_utils import get_device
from src.config.config import Config
import numpy as np

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def transcribe(self, audio_segment: np.ndarray) -> str:
        """
        Transcribes the given audio segment.

        Args:
            audio_segment: The audio segment as a numpy.ndarray.

        Returns:
            The transcription as a string.
        """
        return self.transcribe_segment(audio_segment)

    def transcribe_batch(self, audio_segments: list[np.ndarray], sampling_rate: int = 16000) -> list[str]:
        transcriptions = []
        try:
            inputs = [{"array": audio_segment, "sampling_rate": sampling_rate} for audio_segment in audio_segments]
            
            results = self.asr_model(
                inputs,
                generate_kwargs={
                    "task": "transcribe",
                    "language": "english",
                    "return_timestamps": True,
                }
            )

            for result in results:
                raw_text = result["text"]
                sentences = raw_text.strip().split('. ')
                formatted_text = '. '.join([sentence.capitalize() for sentence in sentences])
                transcriptions.append(formatted_text)

            return transcriptions

        except Exception as e:
            print(f"Error in batch transcription: {e}")
            return [""] * len(audio_segments)
