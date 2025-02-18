import logging
import subprocess
import threading
import time
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OllamaServerManager:
    def __init__(self, log_dir: Path):
        self.process = None
        self.log_file = log_dir / "ollama_server.log"

    def _stream_output(self, stream, log_file):
        """Continuously writes subprocess output to a file."""
        with open(log_file, "a") as f:
            for line in iter(stream.readline, ""):
                f.write(line)
                f.flush()

    def start(self):
        if self.process is not None:
            raise RuntimeError("Ollama server is already running.")

        # Ensure no previous instance is running
        subprocess.run(
            ["ollama", "stop"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )

        self.process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Start threads to stream logs in real-time
        threading.Thread(
            target=self._stream_output,
            args=(self.process.stdout, self.log_file),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._stream_output,
            args=(self.process.stderr, self.log_file),
            daemon=True,
        ).start()

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
