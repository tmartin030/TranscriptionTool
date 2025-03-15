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
    def __init__(self, audio_files, diarizer, transcriber, config, nlp_engine = None):
        self.audio_files = audio_files
        self.diarizer = diarizer
        self.transcriber = transcriber
        self.config = config
        self.nlp_engine = nlp_engine

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

        # Change to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Convert the numpy array to a torch tensor and add a channel dimension (1, time)
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)

        # Create the input dictionary for diarization
        diarization_input = {"waveform": audio_tensor, "sample_rate": sampling_rate}

        # Diarization: now pass the dictionary to the diarizer
        segments = self.diarizer.diarize(diarization_input)

        # Prepare segments in a list
        segment_audios = [
            audio_data[int(start * sampling_rate):int(end * sampling_rate)]
            for start, end, _ in segments
        ]

        # Batch transcription (efficient)
        transcriptions = self.transcriber.transcribe_batch(segment_audios, sampling_rate)
        
        # Associate each transcription with its segment
        transcription_dicts = []
        for (start, end, speaker), transcription in zip(segments, transcriptions):
            # Ensure transcription is always a string
            if isinstance(transcription, list):
                transcription = " ".join(transcription)
            elif not isinstance(transcription, str):
                transcription = str(transcription)
            print(f"Raw transcription: {transcription}")
            if self.nlp_engine:
                cleaned_transcript = self.nlp_engine.clean_transcript(transcription) # Corrected line
                print(f"Cleaned transcription: {cleaned_transcript}")
            else:
                print(f"Skipping nlp cleaning")
                cleaned_transcript = transcription
            transcription_dicts.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "transcription": transcription,
                "cleaned_transcription": cleaned_transcript,
            })

        return audio_path, segments, transcription_dicts
