# Transcription Tool

## Overview

This project is a tool for automatically generating transcripts from audio and video files. It leverages several powerful libraries and models for different stages of the transcription process:

*   **Audio Extraction:** FFmpeg is used to extract audio from video files.
*   **Speaker Diarization:** The `pyannote.audio` library is used to identify different speakers within the audio.
*   **Transcription:** The `transformers` library, specifically the Whisper model (`openai/whisper-large-v3-turbo`), is used to transcribe audio segments into text.
*   **Document Generation:** The `python-docx` library is used to create a well-formatted Word document containing the transcripts, along with a table of contents and clear speaker labels.

## Current Status

The core functionality of the project is working, and it's able to produce transcripts. We are working on improving the quality of the transcripts and the stability of the project.

### Key Components:

*   **`main.py`:** Orchestrates the entire process, from audio extraction to document generation.
*   **`src/config/config.json`:** Contains configuration settings, such as model names and file paths.
*   **`src/dataset/audio_dataset.py`:** Handles loading and preprocessing audio files for the models.
*   **`src/transcription/transcriber.py`:** Manages the loading and usage of the Whisper transcription model.
*   **`src/transcription/diarizer.py`:** Handles loading and using the diarization model.
*   **`src/document/document_generator.py`:** Generates the final transcript document in Word format.
* **`src/utils/gpu_utils.py`**: Handle the GPU connection.

## Challenges and Current Limitations

### Transcription Quality

*   **Room for Improvement:** While the system generates transcripts, the quality is not perfect. There's still room for improvement in terms of accuracy, especially in noisy audio or with multiple speakers. We are working on improving it.

### Testing

*   **Unit Tests Set Aside:** We initially planned to incorporate unit tests, but due to the complexity of the project and the time constraints, we've set aside unit test creation for now. This is an area we intend to revisit later.
*   **Manual Testing:** At the moment, we are manually testing the project.

### CUDA Error and System Sleep/Standby

*   **CUDA Instability:** The project sometimes encounters a `RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable` when the computer wakes from a sleep or standby state.
*   **Workaround:** The current workaround for this issue is to fully restart the application if the computer wakes up from sleep/standby.

## Setup and Usage

### Prerequisites

*   **Python 3.9+**
*   **FFmpeg:** Ensure FFmpeg is installed on your system and available in your `PATH`. You can test it with:
    ```bash
    ffmpeg -version
    ```
* **Hugging Face Token**: You need to create a hugging face token and save it in your environment variable. You can follow [this tutorial](https://huggingface.co/docs/hub/security-tokens).
*   **Dependencies:** Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd TranscriptionTool
    ```
2.  **Set up a Virtual Environment:** It's highly recommended to use a virtual environment.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate   # On Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Add your hugging face token**: Add your hugging face token to your environment variable.
5.  **Configure:** Update `src/config/config.json` with your desired settings:
    *   `diarization_model`: The name of the diarization model.
    *   `asr_model`: The name of the transcription model (Whisper).
    *   `AV_input_dir`: The directory where you'll put your audio/video files.
    *   `transcripts_dir`: The directory where the generated transcripts will be saved.

### Run the project:

1. Put your audio or video files in the directory given by the parameter `AV_input_dir` in your `config.json` file.
2. launch the project from your terminal:
    ```bash
    python main.py
    ```
3. Wait for the end of the processing. The resulting transcripts will be in the directory specified by the parameter `transcripts_dir` in your `config.json` file.

### Add files

* To add video files, simply add it in the `AV_input_dir` folder, it will be automatically converted.
* To add audio files, simply add it in the `AV_input_dir` folder. It should be in `.aac`, `.mp3`, `.wav`, or `.ogg`.

## Future Improvements

*   **Transcription Accuracy:** Investigate techniques to improve the quality of the generated transcripts.
*   **Testing:** Add unit tests to improve the quality and prevent errors.
*   **CUDA Stability:** Address the CUDA stability issue related to sleep/standby.
*   **Performance:** Optimize the code for performance, particularly for very long audio/video files.
* **Error handling**: Improve the error handling in the code.

## Contributing

Contributions are welcome!

## License

[Your License Here (e.g., MIT License)]
