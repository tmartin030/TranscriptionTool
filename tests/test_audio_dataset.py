import unittest
import os
import shutil
import numpy as np
import soundfile as sf
import json
from src.config.config import Config
from unittest.mock import Mock, patch
from src.dataset.audio_dataset import AudioDataset
import torch
import torchaudio



class TestAudioDataset(unittest.TestCase):
    def setUp(self):
        # Set up a test directory and dummy audio files
        self.test_dir = "test_audio_files"
        os.makedirs(self.test_dir, exist_ok=True)
        self.audio_file1 = os.path.join(self.test_dir, "test1.wav")
        self.audio_file2 = os.path.join(self.test_dir, "test2.wav")
        self.empty_audio_file = os.path.join(self.test_dir, "empty.wav")
        self.create_dummy_wav(self.audio_file1)
        self.create_dummy_wav(self.audio_file2)
        self.create_silent_wav(self.empty_audio_file)

        # Create a dummy config file
        self.test_config_path = "test_config.json"
        self.test_config_data = {
            "audio_samples_per_segment": 10000,
        }
        with open(self.test_config_path, "w") as f:
            json.dump(self.test_config_data, f)

        # Create dummy diarizer and transcriber
        self.mock_diarizer = Mock()
        self.mock_transcriber = Mock()

        # Define dummy segments and transcriptions
        self.mock_segments = [(0, 5, "speaker_0")]
        self.mock_transcriptions = ["test transcription 1"]

        # Set the behavior of mock objects
        #self.mock_diarizer.diarize.return_value = self.mock_segments #Removed this line
        self.mock_transcriber.transcribe.return_value = self.mock_transcriptions[0]
        
        # Corrected mock diarizer to check for empty data
        def mock_diarize(audio_data):
            if len(audio_data) == 0:
                return []  # Return empty segments for empty audio
            return self.mock_segments  # Return regular segments otherwise
        self.mock_diarizer.diarize.side_effect = mock_diarize

    def tearDown(self):
        # Clean up the test directory and dummy audio files
        shutil.rmtree(self.test_dir)
        os.remove(self.test_config_path)

    def create_dummy_wav(self, file_path):
        # Create a dummy wav file of 2 seconds
        sampling_rate = 16000
        duration = 2
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        # Correct line: Use soundfile to write WAV
        sf.write(file_path, audio_data, sampling_rate)
    
    def create_silent_wav(self, file_path):
        sample_rate = 16000
        num_frames = sample_rate * 2  # 2 seconds of silence
        waveform = torch.zeros(1, num_frames)  # 1 channel of silence
        torchaudio.save(file_path, waveform, sample_rate)
    
    def create_dataset(self, audio_files, mock_transcriber):
        config = Config(self.test_config_path)
        dataset = AudioDataset(
            audio_files, self.mock_diarizer, mock_transcriber, config
        )
        return dataset

    def test_dataset_length(self):
        audio_files = [self.audio_file1, self.audio_file2]
        dataset = self.create_dataset(audio_files, self.mock_transcriber)
        self.assertEqual(len(dataset), 2)

    def test_getitem(self):
        audio_files = [self.audio_file1]
        dataset = self.create_dataset(audio_files, self.mock_transcriber)
        file_name, segments, transcriptions = dataset[0]
        #corrected this line
        self.assertEqual(file_name, self.audio_file1)
        self.assertEqual(segments, self.mock_segments)
        self.assertEqual(transcriptions, self.mock_transcriptions)
        self.mock_diarizer.diarize.assert_called_once()
        self.mock_transcriber.transcribe.assert_called_once()
    
    def test_empty_audio_file(self):
        audio_files = [self.empty_audio_file]
        config = Config(self.test_config_path)
        dataset = AudioDataset(audio_files, Mock(), Mock(), config)

        #patch audio_dataset to use mock_diarizer and mock_transcriber
        with patch.object(dataset, "diarizer", new=self.mock_diarizer), \
            patch.object(dataset, "transcriber", new=self.mock_transcriber):
            # corrected this line
            file_name, segments, transcriptions = dataset[0]
            self.assertEqual(file_name, self.empty_audio_file)
            self.assertEqual(segments, [])
            self.assertEqual(transcriptions, [])

    def test_audio_padding(self):
        config = Config(self.test_config_path)
        audio_file = os.path.join(self.test_dir, "short.wav")
        # Correct line: Use soundfile to write WAV
        sf.write(audio_file, np.array([0.5] * 100), 16000)
        audio_files = [audio_file]
        dataset = AudioDataset(
            audio_files, self.mock_diarizer, self.mock_transcriber, config
        )
        dataset[0]
        os.remove(audio_file)
