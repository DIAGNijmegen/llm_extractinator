import json
import logging
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

_OLLAMA_HOST = "http://localhost:11434"

#: How long to wait for a server we started to exit before killing it.
_SHUTDOWN_TIMEOUT_SECONDS = 10


def _ollama_is_running(host: str = _OLLAMA_HOST) -> bool:
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


@dataclass(frozen=True)
class ModelInfo:
    """What the server can tell us about a model before it is used.

    Both facts are needed *before* the context window can be sized: whether the
    model thinks decides how much generation budget to reserve, and the native
    context length is the hard ceiling the window cannot exceed. Fetching them
    together keeps them consistent and costs one round trip instead of two.
    """

    supports_thinking: bool = False
    native_context: Optional[int] = None


def _native_context(model_info: Dict[str, Any]) -> Optional[int]:
    """The model's own context length, from ``/api/show``'s ``model_info``.

    The key is namespaced by architecture — ``llama.context_length``,
    ``qwen3.context_length`` — so it is read via ``general.architecture`` rather
    than guessed. The scan afterwards is the fallback for an architecture whose
    namespace we did not anticipate: a right answer from an unexpected key beats
    no answer at all.
    """
    architecture = model_info.get("general.architecture")
    if architecture:
        value = model_info.get(f"{architecture}.context_length")
        if isinstance(value, int) and value > 0:
            return value

    for key, value in model_info.items():
        if key.endswith(".context_length") and isinstance(value, int) and value > 0:
            return value
    return None


def model_capabilities(model_name: str, host: Optional[str] = None) -> ModelInfo:
    """Ask the server about a model. Empty defaults if it cannot be asked.

    Errors are swallowed deliberately. Detection is an *improvement* on the
    user's explicit configuration, not a precondition for it: a server that is
    unreachable, or a model not yet pulled, must not stop a run whose settings
    were given on the command line. ``--reasoning_model`` exists precisely to
    cover the case where this returns nothing useful.
    """
    try:
        data = json.dumps({"name": model_name}).encode()
        req = urllib.request.Request(
            f"{host or _OLLAMA_HOST}/api/show",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            info = json.loads(resp.read())
    except Exception:
        logger.debug("Could not read capabilities for '%s'.", model_name, exc_info=True)
        return ModelInfo()

    return ModelInfo(
        supports_thinking="thinking" in (info.get("capabilities") or []),
        native_context=_native_context(info.get("model_info") or {}),
    )


# A token is at most a couple of dozen characters even in the worst case, so a
# reported count far below chars/20 cannot be a real tokenization of the text.
# prompt_eval_count has a history of being wrong in exactly this direction —
# ollama-python#271 reports it stuck at 1026 across prompts of 57,000 characters
# — and an under-reported count would silently widen the apparent slack, which
# is the one direction that matters here.
_IMPLAUSIBLE_CHARS_PER_TOKEN = 20


def measure_prompt_tokens(
    model_name: str,
    messages: List[Dict[str, str]],
    host: Optional[str] = None,
    num_ctx: Optional[int] = None,
) -> Optional[int]:
    """Ask the model itself how many tokens a prompt is. ``None`` if it cannot.

    Ollama has no tokenize endpoint — #3582 was closed without landing one and
    PR #12030 is still open — but every chat response reports
    ``prompt_eval_count``: the exact number of tokens the server evaluated,
    through the model's own tokenizer *and* its own chat template. Both of those
    are things a local estimate cannot see. Generating a single token is the
    cheapest way to ask the question.

    Returns ``None`` rather than raising on any failure, including an
    implausible answer. A calibration that cannot be performed must not stop a
    run; it just leaves the estimate unverified, which is where things stood
    before this existed.
    """
    try:
        import ollama

        client = ollama.Client(host=host) if host else ollama.Client()
        options = {"num_predict": 1}
        if num_ctx:
            options["num_ctx"] = num_ctx
        response = client.chat(model=model_name, messages=messages, options=options)
        count = (
            response.get("prompt_eval_count")
            if isinstance(response, dict)
            else getattr(response, "prompt_eval_count", None)
        )
    except Exception:
        logger.debug("Could not measure prompt tokens.", exc_info=True)
        return None

    if not isinstance(count, int) or count <= 0:
        return None

    characters = sum(len(message.get("content") or "") for message in messages)
    if count * _IMPLAUSIBLE_CHARS_PER_TOKEN < characters:
        logger.warning(
            "Ollama reported %d prompt tokens for %d characters, which is not a "
            "plausible tokenization; ignoring it (see ollama-python#271).",
            count,
            characters,
        )
        return None
    return count


def model_supports_thinking(model_name: str, host: Optional[str] = None) -> bool:
    """Return True if the given server's model advertises thinking capability."""
    return model_capabilities(model_name, host=host).supports_thinking


class OllamaServerManager:
    def __init__(self, log_dir, host: Optional[str] = None):
        self.process = None
        self.host = host or _OLLAMA_HOST
        # True when the caller pointed us at an already-running server: we only
        # connect to it, never start/pull models onto/stop it ourselves.
        self.externally_managed = host is not None
        self._external = False  # True when Ollama was already running before we started
        self.log_file = log_dir / "ollama_server.log"

    def start_server(self):
        if self.externally_managed:
            logger.info(
                "Using externally managed Ollama server at %s — skipping start.",
                self.host,
            )
            return

        if _ollama_is_running(self.host):
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
            if _ollama_is_running(self.host):
                break
        logger.info("Ollama server started.")

    def stop(self, model_name):
        if self.externally_managed:
            logger.info("Externally managed Ollama server — skipping model stop.")
            return

        command = ["ollama", "stop", model_name]
        try:
            subprocess.run(command, check=True, text=True)
            logger.info(f"Model '{model_name}' stopped successfully.")
        except subprocess.CalledProcessError:
            logger.error(f"Failed to stop model '{model_name}'.")

        self._stop_server()

    def _stop_server(self) -> None:
        """Terminate the server process, if this manager started one.

        ``ollama stop <model>`` unloads the model from VRAM but leaves the
        server running. When we launched that server ourselves it has no other
        owner, so leaving it behind meant every run leaked a process — and on a
        shared machine, one still holding the port.

        A server that was already up when we arrived, or one named explicitly
        with ``--ollama_host``, belongs to somebody else and is left alone;
        ``self.process`` is only set when we started it ourselves.
        """
        if self.process is None:
            return

        logger.info("Stopping the Ollama server this run started.")
        self.process.terminate()
        try:
            self.process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Ollama server did not exit within %ds; killing it.",
                _SHUTDOWN_TIMEOUT_SECONDS,
            )
            self.process.kill()
            try:
                self.process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                logger.error("Ollama server could not be killed; it may be orphaned.")
        finally:
            self.process = None

    def pull_model(self, model_name):
        if self.externally_managed:
            logger.info("Externally managed Ollama server — skipping model pull.")
            return

        command = ["ollama", "pull", model_name]
        try:
            subprocess.run(command, check=True, text=True)
            logger.info(f"Model '{model_name}' pulled successfully.")
        except subprocess.CalledProcessError:
            logger.error(f"Failed to pull model '{model_name}'.")
