import os
from typing import List, Tuple, Any

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map
from pathlib import Path
from src.utils.gpu_utils import get_device
from src.config.config import Config
from pyannote.audio import Model
import yaml

class Diarizer:
    def __init__(self, config: Config, model_dir):
        self.config = config
        self.device = get_device()
        self.model_name = config.get("diarization_model")
        self.diarization_model_dir = model_dir
        self.nemo_model_dir = config.get("nemo_model_dir")
        self.temp_dir = config.get("temp_dir")
        self.hf_token = os.environ.get("HF_TOKEN")  # Get the token from the environment variable
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable not set. Please set it to your Hugging Face token.")

        if not os.path.exists(self.diarization_model_dir):
            os.makedirs(self.diarization_model_dir)

        model_path = os.path.join(self.diarization_model_dir, self.model_name.replace("/", "-"))
        model_path_obj = Path(model_path)
        if not os.path.exists(model_path):
            print(f"Downloading {self.model_name} to {self.diarization_model_dir}...")
            self.pipeline = SpeakerDiarization.from_pretrained(self.model_name, use_auth_token=self.hf_token).to(self.device)
            self.pipeline.dump_params(model_path_obj)
        else:
            print(f"Loading existing {self.model_name} from {self.diarization_model_dir}...")
            if not os.path.exists(os.path.join(model_path, "config.yml")) or not os.path.exists(os.path.join(model_path, "params.yml")):
                self.pipeline = SpeakerDiarization.from_pretrained(self.model_name, use_auth_token=self.hf_token).to(self.device)
                self.pipeline.dump_params(model_path_obj)
            else:
                self.pipeline = SpeakerDiarization.from_pretrained(model_path, use_auth_token=self.hf_token).to(self.device)
                self.pipeline.load_params(model_path_obj)
        # self.nemo_model = self.load_nemo_model()

    def diarize(self, audio_input: Any) -> List[Tuple[float, float, str]]:
        """
        Performs speaker diarization on the given audio input.

        Args:
            audio_input: The audio input, which can be:
                - a file path string or Path,
                - an IOBase instance,
                - a Mapping with an "audio" key,
                - or a Mapping with both "waveform" and "sample_rate" keys.

        Returns:
            A list of tuples, each containing:
                - start time (float),
                - end time (float), and
                - the speaker label (str).
        """
        try:
            diarization = self.pipeline(audio_input)
            diarization_result = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                # Extract start and end times from the segment.
                diarization_result.append((segment.start, segment.end, speaker))
            return diarization_result
        except Exception as e:
            print(f"Error in diarization with pyannote model {audio_input}: {e}")
            raise Exception(f"Error in diarization and nemo model not loaded {audio_input}: {e}")

    def load_nemo_model(self):
        try:
            print(f"Loading NeMo Neural Diarizer")
            manifest = os.path.join(self.temp_dir, "manifest.json")
            rttm_dir = os.path.join(self.temp_dir, "rttm")
            os.makedirs(rttm_dir, exist_ok=True)
            with open(manifest, 'w') as fp:
                pass  # Create an empty manifest file
            model_path = os.path.join(self.nemo_model_dir, "diar_msdd_telephonic")
            if not os.path.exists(model_path):
                print("Downloading NeMo Neural Diarizer...")
                msdd_model = NeuralDiarizer.from_pretrained(model_name="diar_msdd_telephonic")
                msdd_model.save_to(model_path)
            else:
                print(f"Loading existing NeMo Neural Diarizer from {model_path}...")
                msdd_model = NeuralDiarizer.restore_from(model_path)
            msdd_model.set_manifest_filepath(manifest)
            msdd_model.set_rttm_dir(rttm_dir)
            return msdd_model
        except Exception as e:
            print(f"Error in loading nemo diarization model: {e}")
            return None

    def diarize_with_nemo(self, audio_file: str):
        try:
            self.nemo_model.diarize_audio(audio_file)
            rttm_files = os.listdir(os.path.join(self.temp_dir, "rttm"))
            if not rttm_files:
                raise Exception(f"No rttm files were found in nemo diarization")
            rttm_file = os.path.join(self.temp_dir, "rttm", rttm_files[0])
            rttm_mapping = audio_rttm_map(rttm_file)
            diarization_result = []
            for speaker_id, segments in rttm_mapping.items():
                for segment in segments:
                    diarization_result.append((segment, speaker_id))
            return diarization_result
        except Exception as e:
            print(f"Error in diarization with nemo model {audio_file}: {e}")
            return []
