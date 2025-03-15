import os
import torch
from transformers import pipeline
from src.config.config import Config

class NLPEngine:
    def __init__(self, config: Config, model_dir):
        self.config = config
        self.device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU
        self.model_name = config.get("nlp_model")
        self.model_dir = model_dir
        self.cleaning_prompt = config.get("nlp_cleaning_prompt")
        self.summary_prompt = config.get("nlp_summary_prompt")
        # Define the maximum input length for the model
        self.max_input_length = 1024
        # Define the maximum output length for the model
        self.max_output_length = 100
        # Define the minimum output length for the model
        self.min_output_length = 20

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        model_path = os.path.join(self.model_dir, self.model_name.replace("/", "-"))

        if not os.path.exists(model_path):
            print(f"Downloading {self.model_name} to {self.model_dir}...")
            self.model = pipeline(
                "text2text-generation", model=self.model_name, device=self.device
            )
            self.model.save_pretrained(model_path)
        else:
            print(f"Loading existing {self.model_name} from {self.model_dir}...")
            self.model = pipeline(
                "text2text-generation", model=model_path, device=self.device
            )
            
    def clean_transcript(self, transcript: str) -> str:
        """
        Uses the NLP model to clean the transcript by removing filler words, unwanted punctuation,
        and formatting extraneous speech.
        """
        # Truncate the input to the maximum input length
        transcript = transcript[:self.max_input_length]
        prompt = f"{self.cleaning_prompt}\nTranscript: {transcript}\nCleaned Transcript:"
        try:
            result = self.model(prompt, max_length=self.max_output_length, truncation=True)
            return result[0]['generated_text']
        except Exception as e:
            print(f"Error during cleaning: {e}")
            return transcript  # Return the original transcript on error

    def generate_summary(self, cleaned_transcript: str) -> str:
        """
        Uses the NLP model to generate a summary of the cleaned transcript.
        """
        prompt = f"{self.summary_prompt}\nTranscript: {cleaned_transcript}\nSummary:"
        try:
            result = self.model(prompt, max_length=self.max_output_length, min_length=self.min_output_length, do_sample=False)
            return result[0]['generated_text']
        except Exception as e:
            print(f"Error during summarization: {e}")
            return "Summary generation failed."
