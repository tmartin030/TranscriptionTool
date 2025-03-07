import os
import subprocess
import shutil
from src.config.config import Config

class Preprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.temp_dir = config.get("temp_dir")
        self.sample_rate = 16000

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            

    def convert_to_wav(self, input_file: str, output_file: str):
        """Converts any audio/video file to WAV."""
        try:
            command = [
                "ffmpeg",
                "-i", input_file,
                "-vn",  # Disable video
                "-ac", "1",  # Mono audio
                "-ar", str(self.sample_rate),  # Sample rate
                "-sample_fmt", "s16",
                output_file,
                "-y" # overwrite output file
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Converted to WAV: {output_file}")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error converting {input_file} to WAV: {e.stderr}")

    def filter_audio(self, input_file: str, output_file: str):
        """Applies high-pass and low-pass filters."""
        try:
            command = [
                "ffmpeg",
                "-i", input_file,
                "-af", f"highpass=f=200,lowpass=f=3000",  # Example: 200Hz high-pass, 3000Hz low-pass
                output_file,
                "-y"
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Filtered audio: {output_file}")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error filtering audio {input_file}: {e.stderr}")

    def isolate_vocals(self, input_file: str, output_file: str):
        """Isolates vocals using a tool like Demucs."""
        try:
            # Assuming Demucs is installed and in PATH
            command = [
                "python3", "-m", "demucs",
                "--two-stems=vocals",
                "--out", os.path.dirname(output_file),
                input_file,
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            # Demucs outputs to a subdirectory, so we need to move the file
            stem_dir = os.path.join(os.path.dirname(output_file), "htdemucs", os.path.splitext(os.path.basename(input_file))[0])
            vocal_file = os.path.join(stem_dir, "vocals.wav")
            shutil.move(vocal_file, output_file)
            print(f"Isolated vocals: {output_file}")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error isolating vocals {input_file}: {e.stderr}")
        except FileNotFoundError:
            raise Exception(f"Demucs not found, please install demucs.")

    def preprocess(self, input_file: str, vocal_target: bool = True) -> str:
        """
        Applies all preprocessing steps to the audio file.

        Args:
            input_file: Path to the input audio/video file.
            vocal_target: If we want the vocal target file or not

        Returns:
            The path to the preprocessed audio file.
        """
        try:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            wav_file = os.path.join(self.temp_dir, f"{base_name}.wav")
            filtered_file = os.path.join(self.temp_dir, f"{base_name}_filtered.wav")
            vocal_file = os.path.join(self.temp_dir, f"{base_name}_vocals.wav")

            self.convert_to_wav(input_file, wav_file)
            self.filter_audio(wav_file, filtered_file)
            if vocal_target:
                self.isolate_vocals(filtered_file, vocal_file)
                return vocal_file
            else:
                return filtered_file

        except Exception as e:
            raise Exception(f"Error in preprocessing: {e}")
