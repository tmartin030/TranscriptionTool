import unittest
import os
import sys
import shutil
import tempfile
from unittest.mock import patch, Mock
from src.config.config import Config
import subprocess
import json
from main import main
from src.dataset.audio_dataset import AudioDataset  # Import AudioDataset
from src.transcription.diarizer import Diarizer
from src.transcription.transcriber import Transcriber
import torch
import torchaudio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestMain(unittest.TestCase):
    def setUp(self):
        # Set up a test directory
        self.test_dir = tempfile.mkdtemp()

        # Set up a dummy config file
        self.test_config_path = os.path.join(self.test_dir, "config.json")
        self.test_config_data = {
            "AV_input_dir": os.path.join(self.test_dir, "AV_input_dir"),
            "temp_dir": os.path.join(self.test_dir, "temp_dir"),
            "transcripts_dir": os.path.join(self.test_dir, "transcripts_dir"),
            "diarization_model_path": "/path/to/diarization_model",  # Dummy paths
            "transcription_model_path": "/path/to/transcription_model",
        }

        os.makedirs(self.test_config_data["AV_input_dir"])
        os.makedirs(self.test_config_data["temp_dir"])
        os.makedirs(self.test_config_data["transcripts_dir"])

        with open(self.test_config_path, "w") as f:
            json.dump(self.test_config_data, f)

        # Create dummy video and audio files
        self.video_file = os.path.join(
            self.test_config_data["AV_input_dir"], "test_video.mp4"
        )
        self.audio_file = os.path.join(
            self.test_config_data["AV_input_dir"], "test_audio.wav"
        )
        self.create_dummy_file(self.video_file)
        self.create_dummy_file(self.audio_file)

    def tearDown(self):
        # Clean up the test directory
        shutil.rmtree(self.test_dir)

    def create_dummy_file(self, file_path):
        sample_rate = 16000
        num_frames = sample_rate * 2  # 2 seconds of silence
        waveform = torch.zeros(1, num_frames)  # 1 channel of silence
        torchaudio.save(file_path, waveform, sample_rate)
    @patch("subprocess.run")
    def test_extract_audio(self, mock_subprocess_run):
        config = Config(self.test_config_path)
        video_path = os.path.join(config.get("AV_input_dir"), "test_video.mp4")
        audio_path = os.path.join(config.get("temp_dir"), "test_video.aac")

        # Define expected behavior of mock_subprocess_run
        mock_subprocess_run.return_value = Mock(returncode=0, stdout="", stderr="")

        from main import extract_audio

        extract_audio(video_path, audio_path)
        mock_subprocess_run.assert_called_once()

    @patch.object(Transcriber, "__init__", return_value=None) # Mock Transcriber.__init__
    @patch.object(Diarizer, "__init__", return_value=None) # Mock Diarizer.__init__
    @patch.object(Transcriber, "transcribe", return_value= "test transcription") # Mock Transcriber.transcribe
    @patch.object(Diarizer, "diarize", return_value = [(0, 5, "speaker_0")]) # Mock Diarizer.diarize
    @patch("subprocess.run")
    def test_main(self, mock_subprocess_run, mock_diarizer, mock_transcriber, mock_diarizer_init, mock_transcriber_init):
        # Mock the behavior of subprocess.run for the video extraction
        mock_subprocess_run.return_value = Mock(returncode=0, stdout="", stderr="")

        config = Config(self.test_config_path)
        # Create dummy Dataset
        # Mock audio files to be a list containing a file path
        audio_files_mock = [self.audio_file]
        
        audio_dataset = AudioDataset(audio_files=audio_files_mock,
                       diarizer=Mock(),
                       transcriber=Mock(),
                       config=config)
        # Run the main function
        #Patch AudioDataset.__getitem__ to return a known value
        with patch.object(AudioDataset, '__getitem__', return_value = (os.path.basename(self.audio_file), [(0, 5, "speaker_0")], ["test transcription"])):
            main()

        # Assert that the main function ran without errors
        self.assertTrue(True)  # If no exceptions were raised, the test passes

if __name__ == "__main__":
    unittest.main()
