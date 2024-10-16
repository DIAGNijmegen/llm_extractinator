import subprocess
import os

class OllamaServerManager:
    def __init__(self, model_name, log_filename="ollama_server.log"):
        """
        Initialize the server manager with the given model name.
        """
        self.model_name = model_name
        self.log_directory = os.path.join(os.path.dirname(__file__), '../output', self.model_name)
        self.log_file_path = os.path.join(self.log_directory, log_filename)
        self.serve_process = None

        # Ensure the output directory exists
        os.makedirs(self.log_directory, exist_ok=True)

    def pull_model(self):
        """
        Pull the specified model using the `ollama pull` command.
        """
        pull_command = f"ollama pull {self.model_name}"
        print(f"Pulling model: {self.model_name}...")
        pull_process = subprocess.Popen(pull_command, shell=True)
        pull_process.wait()  # Wait for the pull command to complete
        print(f"Model {self.model_name} pulled successfully.")

    def start_server(self):
        """
        Start the server for the specified model using the `ollama serve` command.
        """
        log_file_handle = open(self.log_file_path, "w")
        
        serve_command = f"ollama serve"
        print(f"Starting server...")
        self.serve_process = subprocess.Popen(
            serve_command, 
            shell=True,
            stdout=log_file_handle,
            stderr=subprocess.STDOUT
        )
        print("Ollama server is running...")

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
