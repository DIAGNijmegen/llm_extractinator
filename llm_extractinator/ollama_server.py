import os
import subprocess
from pathlib import Path

import ollama


class OllamaServerManager:
    def __init__(self, model_name, log_dir, log_filename="ollama_server.log"):
        """
        Initialize the server manager with the given model name.
        """
        self.model_name = model_name
        self.log_dir = log_dir
        self.log_file_path = self.log_dir / log_filename
        self.serve_process = None

        # Ensure the output directory exists
        os.makedirs(self.log_dir, exist_ok=True)

    def pull_model(self):
        """
        Pull the specified model using the `ollama pull` command.
        """
        try:
            print(f"Pulling model: {self.model_name}...")
            ollama.pull(self.model_name)
            print(f"Model {self.model_name} pulled successfully.")
        except Exception as e:
            print(f"Error pulling model {self.model_name}: {e}")

    def start_server(self):
        """
        Start the server for the specified model using the `ollama serve` command.
        """
        log_file_handle = open(self.log_file_path, "w")

        try:
            serve_command = f"ollama serve"
            print(f"Starting server...")
            self.serve_process = subprocess.Popen(
                serve_command,
                shell=True,
                stdout=log_file_handle,
                stderr=subprocess.STDOUT,
            )
            print("Ollama server is running...")
        except Exception as e:
            print(f"Error starting Ollama server: {e}")
            log_file_handle.close()

    def stop_server(self):
        """
        Stop the server if it is running.
        """
        if self.serve_process:
            print("Terminating Ollama server...")
            self.serve_process.terminate()
            self.serve_process.wait()  # Ensure the process has been terminated
            print("Ollama server terminated.")
            self.serve_process = None

    def __enter__(self):
        """
        Context manager entry point.
        """
        # Pull the model and start the server
        self.start_server()
        self.pull_model()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager exit point.
        Stops the server if the script exits or crashes.
        """
        # Stop the server if the script exits or crashes
        self.stop_server()
