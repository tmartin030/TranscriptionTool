import os
import librosa

class AudioSplitter:
    @staticmethod
    def split_audio(input_file: str, output_dir: str, segment_length: int = 60, overlap: int = 10) -> list[str]:
        """
        Splits an audio file into segments of the specified length with overlap.

        Args:
            input_file: Path to the input audio file.
            output_dir: Directory to save the split audio segments.
            segment_length: Length of each segment in seconds.
            overlap: Overlap between segments in seconds.

        Returns:
            A list of paths to the generated audio segment files.
        """
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            y, sr = librosa.load(input_file, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            segment_files = []
            start_time = 0
            while start_time < duration:
                end_time = min(start_time + segment_length, duration)
                segment_file = os.path.join(output_dir, f"segment_{start_time:.2f}_{end_time:.2f}.wav")
                AudioSplitter.save_segment(y, sr, start_time, end_time, segment_file)
                segment_files.append(segment_file)
                start_time += (segment_length - overlap)
            return segment_files
        except Exception as e:
            raise Exception(f"Error in splitting {input_file}: {e}")

    @staticmethod
    def save_segment(y, sr, start_time, end_time, output_file):
        """Saves a segment of audio."""
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        librosa.write_wav(output_file, y[start_sample:end_sample], sr)  # Changed line
