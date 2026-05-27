import json
import logging
import subprocess
import time
import urllib.error
import urllib.request

# Configure logging
logger = logging.getLogger(__name__)

_OLLAMA_HOST = "http://localhost:11434"
_OLLAMA_URL = f"{_OLLAMA_HOST}/api/tags"


def _ollama_is_running() -> bool:
    try:
        urllib.request.urlopen(_OLLAMA_URL, timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


def model_supports_thinking(model_name: str) -> bool:
    """Return True if the locally installed model advertises thinking capability."""
    try:
        data = json.dumps({"name": model_name}).encode()
        req = urllib.request.Request(
            f"{_OLLAMA_HOST}/api/show",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            info = json.loads(resp.read())
        return "thinking" in info.get("capabilities", [])
    except Exception:
        return False


class OllamaServerManager:
    def __init__(self, log_dir):
        self.process = None
        self._external = False  # True when Ollama was already running before we started
        self.log_file = log_dir / "ollama_server.log"

    def start_server(self):
        if _ollama_is_running():
            logger.info("Ollama server already running — skipping start.")
            self._external = True
            return

        if self.process is not None:
            raise RuntimeError("Ollama server is already running.")

        with open(self.log_file, "w") as log:
            self.process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=log,
                stderr=log,
                text=True,
            )

        # Wait for the server to become ready
        for _ in range(10):
            time.sleep(1)
            if _ollama_is_running():
                break
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
