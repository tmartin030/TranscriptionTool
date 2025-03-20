import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment
from pyannote.core import Annotation
from pyannote.core import notebook
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
import numpy as np

class Diarizer:
    def __init__(self, config, model_dir):
        self.config = config
        self.model_dir = model_dir
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", cache_dir=model_dir)
        self.pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.pipeline.min_duration_on = 1 # Changed min_duration_on

    def diarize(self, file_name_item):
        if isinstance(file_name_item, dict):
            file_name = file_name_item.get("audio")
        elif isinstance(file_name_item, str):
            file_name = file_name_item
        else:
            raise ValueError(f"Invalid file name item: {file_name_item}")
        
        diarization = self.pipeline(file_name)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        return segments
