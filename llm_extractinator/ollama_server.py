import logging
import subprocess
import time

# Configure logging
logger = logging.getLogger(__name__)


class OllamaServerManager:
    def __init__(self):
        self.process = None

    def start(self):
        if self.process is not None:
            raise RuntimeError("Ollama server is already running.")

        self.process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.process is None:
            logger.warning("Ollama server is not running.")
            return

        self.process.terminate()
        self.process.wait()
        self.process = None
        logger.info("Ollama server stopped.")
