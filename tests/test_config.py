import unittest
import json
import os
from src.config.config import Config

class TestConfig(unittest.TestCase):
    def setUp(self):
        """Create a temporary config file for testing."""
        self.test_config_path = "test_config.json"
        self.test_config_data = {
            "AV_input_dir": "test_audio_input",
            "temp_dir": "test_temp",
            "transcripts_dir": "test_transcripts",
            "key1": "value1",
            "key2": 123,
        }
        with open(self.test_config_path, "w") as f:
            json.dump(self.test_config_data, f)

    def tearDown(self):
        """Remove the temporary config file after testing."""
        os.remove(self.test_config_path)

    def test_load_config(self):
        """Test that the config loads correctly."""
        config = Config(self.test_config_path)
        self.assertEqual(config.data, self.test_config_data)

    def test_get_existing_key(self):
        """Test getting an existing key."""
        config = Config(self.test_config_path)
        self.assertEqual(config.get("key1"), "value1")

    def test_get_nonexistent_key_with_default(self):
        """Test getting a nonexistent key with a default."""
        config = Config(self.test_config_path)
        self.assertEqual(config.get("nonexistent_key", "default_value"), "default_value")

    def test_get_nonexistent_key_without_default(self):
        """Test getting a nonexistent key without a default."""
        config = Config(self.test_config_path)
        self.assertIsNone(config.get("nonexistent_key"))

    def test_file_not_found(self):
        """Test that a FileNotFoundError is raised for a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            Config("nonexistent_config.json")

    def test_invalid_json(self):
        """Test that a ValueError is raised for an invalid JSON file."""
        with open("invalid_config.json", "w") as f:
            f.write("this is not json")
        with self.assertRaises(ValueError):
            Config("invalid_config.json")
        os.remove("invalid_config.json")


if __name__ == "__main__":
    unittest.main()
