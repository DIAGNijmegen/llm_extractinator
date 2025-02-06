import subprocess
import time


class OllamaServerManager:
    def __init__(self, host="localhost", port=28900):
        """
        Initializes the OllamaServerManager.
        :param host: Host address to bind the server.
        :param port: Port to run the server.
        """
        self.host = host
        self.port = port
        self.process = None

    def start(self):
        """
        Starts the Ollama server process.
        """
        if self.process is not None:
            raise RuntimeError("Ollama server is already running.")

        command = ["ollama", "serve"]
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for the server to be fully up
        timeout = 15  # Maximum wait time
        elapsed = 0
        while not self.is_server_running():
            time.sleep(1)
            elapsed += 1
            if elapsed > timeout:
                raise RuntimeError("Ollama server failed to start within 15 seconds.")
        print("Ollama server started.")

    def stop(self, model_name):
        """
        Stops the Ollama server process manually.
        """
        command = ["ollama", "stop", model_name]

        try:
            subprocess.run(command, check=True, text=True)
            print(f"Model '{model_name}' stopped successfully.")
        except subprocess.CalledProcessError:
            print(f"Failed to stop model '{model_name}'.")

    def pull_model(self, model_name):
        """
        Pulls a specified model for offline use.
        :param model_name: Name of the model to pull.
        """
        command = ["ollama", "pull", model_name]
        try:
            subprocess.run(command, check=True, text=True)
            print(f"Model '{model_name}' pulled successfully.")
        except subprocess.CalledProcessError:
            print(f"Failed to pull model '{model_name}'.")

    def __enter__(self):
        """Starts the server when entering the context."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stops the server automatically when exiting the context."""
        if self.process is None:
            print("No Ollama server is running.")
            return

        self.process.terminate()
        self.process.wait()
        self.process = None
        print("Ollama server stopped.")
