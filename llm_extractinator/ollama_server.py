import logging
import subprocess
import time

# Configure logging
logger = logging.getLogger(__name__)


class OllamaServerManager:
    def __init__(self, log_dir):
        self.process = None
        self.log_file = log_dir / "ollama_server.log"

    def start_server(self):
        if self.process is not None:
            raise RuntimeError("Ollama server is already running.")

        with open(self.log_file, "w") as log:
            self.process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=log,
                stderr=log,
                text=True,
            )

            # Wait for the server to start
            time.sleep(5)
            logger.info("Ollama server started.")

    def stop(self, model_name):
        command = ["ollama", "stop", model_name]
        try:
            subprocess.run(command, check=True, text=True)
            logger.info(f"Model '{model_name}' stopped successfully.")
        except subprocess.CalledProcessError:
            logger.error(f"Failed to stop model '{model_name}'.")

    def pull_model(self, model_name):
        command = ["ollama", "pull", model_name]
        try:
            subprocess.run(command, check=True, text=True)
            logger.info(f"Model '{model_name}' pulled successfully.")
        except subprocess.CalledProcessError:
            logger.error(f"Failed to pull model '{model_name}'.")
