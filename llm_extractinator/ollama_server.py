import subprocess
import time
import socket

class OllamaServerManager:
    def __init__(self, host="localhost", port=28900):
        self.host = host
        self.port = port
        self.process = None

    def is_server_running(self):
        """Check if the Ollama server is running and listening on the port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, self.port)) == 0

    def start(self):
        if self.process is not None:
            raise RuntimeError("Ollama server is already running.")

        self.process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait for the server to be fully up
        timeout = 15  # Maximum wait time
        elapsed = 0
        while not self.is_server_running():
            time.sleep(1)
            elapsed += 1
            if elapsed > timeout:
                raise RuntimeError("Ollama server failed to start within 15 seconds.")

        print("Ollama server started and ready.")

    def stop(self, model_name):
        command = ["ollama", "stop", model_name]
        try:
            subprocess.run(command, check=True, text=True)
            print(f"Model '{model_name}' stopped successfully.")
        except subprocess.CalledProcessError:
            print(f"Failed to stop model '{model_name}'.")

    def pull_model(self, model_name):
        command = ["ollama", "pull", model_name]
        try:
            subprocess.run(command, check=True, text=True)
            print(f"Model '{model_name}' pulled successfully.")
        except subprocess.CalledProcessError:
            print(f"Failed to pull model '{model_name}'.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.process is None:
            print("No Ollama server is running.")
            return

        self.process.terminate()
        self.process.wait()
        self.process = None
        print("Ollama server stopped.")
