"""The Studio's run settings, and how they become an ``extractinate`` command.

Kept out of ``gui.py`` deliberately: that module runs Streamlit at import time,
so anything living there cannot be unit tested. Everything here is pure — build
a :class:`RunSettings`, ask it for a command line or a summary, assert on the
result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from llm_extractinator.budget import REASONING_ALLOWANCE
from llm_extractinator.output_parsers import estimate_output_tokens


class InvalidTaskName(ValueError):
    """Raised when a task filename has no ``TaskNNN`` prefix to read an id from."""


_TASK_ID_RE = re.compile(r"Task(\d{3})")


def task_id_from_filename(name: str) -> str:
    """Pull the numeric id out of a ``TaskNNN_something.json`` filename."""
    match = _TASK_ID_RE.match(name)
    if match is None:
        raise InvalidTaskName(
            f"{name!r} does not start with a three-digit task id (e.g. Task001_…)"
        )
    return match.group(1)


@dataclass
class RunSettings:
    """Everything the Run tab collects, in CLI terms rather than widget terms."""

    task_file: str
    model_name: str

    # None means "say nothing, let the backend auto-detect"; True/False are the
    # explicit --reasoning_model / --no_reasoning overrides.
    reasoning: Optional[bool] = None

    n_runs: int = 1
    verbose: bool = False
    overwrite: bool = False
    seed: Optional[int] = None

    chunk_size: Optional[int] = None
    test_run_size: Optional[int] = None
    test_run_random: bool = False

    num_examples: int = 0
    max_context_len: str = "max"
    max_context_cap: Optional[int] = None

    # None means auto: greedy for an ordinary model, 0.6 for a thinking one.
    temperature: Optional[float] = None

    # None means auto, exactly as it does on ``TaskConfig``: the backend sizes
    # the budget from the output schema and adds the reasoning allowance itself.
    # This used to be ``int = 512``, with 512 doubling as the "do not emit the
    # flag" sentinel — so the Studio showed 512, sent nothing, and the run used
    # a schema-derived number that was neither. One value, three meanings.
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    ollama_host: Optional[str] = None

    def to_command(self) -> list[str]:
        """The exact ``extractinate`` invocation these settings describe.

        Only non-default values are emitted, so the command shown to the user
        stays readable and copy-pasteable.
        """
        cmd = [
            "extractinate",
            "--task_id",
            task_id_from_filename(self.task_file),
            "--model_name",
            self.model_name,
        ]
        if self.reasoning is True:
            cmd.append("--reasoning_model")
        elif self.reasoning is False:
            cmd.append("--no_reasoning")
        if self.n_runs != 1:
            cmd += ["--n_runs", str(self.n_runs)]
        if self.verbose:
            cmd.append("--verbose")
        if self.overwrite:
            cmd.append("--overwrite")
        if self.seed is not None:
            cmd += ["--seed", str(self.seed)]
        if self.chunk_size is not None:
            cmd += ["--chunk_size", str(self.chunk_size)]
        if self.test_run_size is not None:
            cmd += ["--test_run_size", str(self.test_run_size)]
            if self.test_run_random:
                cmd.append("--test_run_random")
        if self.temperature is not None:
            cmd += ["--temperature", str(self.temperature)]
        if self.top_k is not None:
            cmd += ["--top_k", str(self.top_k)]
        if self.top_p is not None:
            cmd += ["--top_p", str(self.top_p)]
        if self.num_predict is not None:
            cmd += ["--num_predict", str(self.num_predict)]
        if self.max_context_len != "max":
            cmd += ["--max_context_len", self.max_context_len]
        if self.max_context_cap is not None:
            cmd += ["--max_context_cap", str(self.max_context_cap)]
        if self.num_examples:
            cmd += ["--num_examples", str(self.num_examples)]
        if self.ollama_host:
            cmd += ["--ollama_host", self.ollama_host]
        return cmd

    @property
    def scope_label(self) -> str:
        """One phrase for how much of the dataset this run touches."""
        if self.test_run_size is None:
            base = "Full dataset"
        else:
            how = "random" if self.test_run_random else "first"
            base = f"Test run · {how} {self.test_run_size} rows"
        if self.chunk_size is not None:
            base += f" · chunks of {self.chunk_size}"
        return base

    @property
    def context_label(self) -> str:
        """One phrase for the context-window strategy, cap included."""
        if self.max_context_len == "max":
            base = "Fitted to data"
        elif self.max_context_len == "split":
            base = "Split short/long"
        else:
            base = f"{self.max_context_len} tokens"
        if self.max_context_cap is not None:
            base += f" · max {self.max_context_cap}"
        return base

    @property
    def temperature_label(self) -> str:
        if self.temperature is not None:
            return str(self.temperature)
        return f"Auto ({THINKING_TEMPERATURE} thinking / {GREEDY_TEMPERATURE} otherwise)"

    @property
    def reasoning_label(self) -> str:
        return {True: "On", False: "Off", None: "Auto-detect"}[self.reasoning]

    @property
    def num_predict_label(self) -> str:
        """What the run will actually do, rather than a number that isn't used.

        Auto is the common case and the review strip has to say so: printing a
        figure here implies the run is pinned to it, which is exactly the
        confusion this label exists to end.
        """
        if self.num_predict is None:
            return "Auto (from schema)"
        return f"{self.num_predict:,}"

    def summary(self) -> list[tuple[str, str, bool]]:
        """Label/value/highlight triples for the review strip above **Run**.

        The highlight flag marks settings that differ from a plain full run, so
        the things most likely to surprise someone are the things that stand out.
        """
        return [
            ("Model", self.model_name, True),
            ("Scope", self.scope_label, self.test_run_size is not None),
            ("Context", self.context_label, self.max_context_cap is not None),
            ("Reasoning", self.reasoning_label, self.reasoning is not None),
            ("Max tokens", self.num_predict_label, self.num_predict is not None),
            ("Temperature", self.temperature_label, self.temperature is not None),
        ]


@dataclass(frozen=True)
class HardwarePreset:
    """A model and context ceiling sized for a VRAM budget.

    Two things are deliberately *not* here:

    ``reasoning`` — the backend pulls the model before inspecting it, so
    auto-detection is already right at run time and a fixed value could only be
    wrong.

    ``num_predict`` — output length follows the schema and whether the model
    thinks, not the size of the card. A 5-20 field JSON object is 80-350 tokens
    whatever GPU produces it.
    """

    model: str
    context_cap: int
    min_vram_gb: int
    note: str

    def describe(self) -> str:
        return f"{self.model} · {self.context_cap:,}-token context ceiling"


# Context ceilings are computed, not guessed: weights at Q4_K_M plus an f16 KV
# cache of 2 * layers_with_kv * kv_heads * head_dim * 2 bytes per token, fitted
# into 85% of the card.
#
# The counter-intuitive one is High. qwen3.5:35b is a 35B-A3B MoE with hybrid
# Gated DeltaNet attention (full_attention_interval=4), so only 10 of its 40
# layers keep a KV cache — 20 KiB/token against phi4's 200 KiB/token. Context is
# nearly free on it; the 22.4 GiB of weights is the whole constraint, which is
# also why it does not fit a 24GB card at all. Its ceiling is the model's native
# maximum rather than anything to do with VRAM.
#
# Low permitting a *larger* window than Medium is likewise real, not a typo:
# qwen3:4b leaves far more of its card free than phi4:14b does.
HARDWARE_PRESETS: dict[str, HardwarePreset] = {
    "High — 40GB+ VRAM": HardwarePreset(
        model="qwen3.5:35b",
        context_cap=262_144,
        min_vram_gb=40,
        note=(
            "Best extraction quality. Needs 40GB — the weights alone are 22.4GiB, "
            "so this will not fit a 24GB card. Its hybrid attention makes context "
            "cheap, so the ceiling here is the model's native 262K limit rather "
            "than a memory one."
        ),
    ),
    "Medium — 12GB VRAM": HardwarePreset(
        model="phi4:14b",
        context_cap=8192,
        min_vram_gb=12,
        note=(
            "A good default, and the one to start from. Fits a 12GB consumer card "
            "such as a 3060 or 4070, where the 8.5GiB of weights leave room for "
            "roughly 9,000 tokens of context."
        ),
    ),
    "Low — 8GB VRAM": HardwarePreset(
        model="qwen3:4b",
        context_cap=16384,
        min_vram_gb=8,
        note=(
            "Fast and undemanding, but expect noticeably lower extraction quality. "
            "Small weights leave plenty of room for context — more than Medium, "
            "despite the smaller card."
        ),
    ),
}

CUSTOM_PRESET = "Custom"


# Qwen's model cards are explicit that a thinking model must not run greedy:
# "DO NOT use greedy decoding, as it can lead to performance degradation and
# endless repetitions." A repetition loop is a particularly bad failure here — it
# spends the whole num_predict budget and returns no JSON at all. 0.6 is the
# temperature Qwen recommends for thinking mode (alongside top_p 0.95, top_k 20).
# Determinism is not lost: pair it with --seed.
THINKING_TEMPERATURE = 0.6
GREEDY_TEMPERATURE = 0.0


def resolved_temperature(
    temperature: Optional[float], reasoning: Optional[bool]
) -> float:
    """The temperature a run will actually use, given an explicit value or None."""
    if temperature is not None:
        return temperature
    return THINKING_TEMPERATURE if reasoning else GREEDY_TEMPERATURE


def resolve_thinking(
    detected: bool, reasoning_model: bool, no_reasoning: bool
) -> bool:
    """Whether this run is treated as reasoning.

    Shared between ``TaskRunner`` (which needs it before sizing the window) and
    ``PredictionTask`` (which needs it for the ``think`` parameter and the
    temperature). One expression, so the two cannot disagree about a run.

    ``no_reasoning`` wins over everything. ``reasoning_model`` exists only for
    what detection cannot see — a model not yet pulled, or an unreachable
    server — since detection answers False on any error.
    """
    if no_reasoning:
        return False
    return bool(detected) or bool(reasoning_model)


#: The floor for an automatically chosen output budget. A 5-20 field JSON
#: object runs 80-350 tokens, so this leaves comfortable headroom for anything
#: small — and flooring here means auto-sizing can only ever *raise* the budget
#: relative to what this project shipped before, never lower it.
DEFAULT_NUM_PREDICT = 512


def suggested_num_predict(
    reasoning: Optional[bool] = None, parser_model=None
) -> int:
    """The output budget for a run that did not specify one.

    ``--num_predict`` was a number every user had to invent, and one value had
    to serve a schema of three enum fields and a schema of thirty free-text
    ones. Deriving it from the schema makes it an override rather than a
    requirement — the same shape as ``--temperature`` and the reasoning flags,
    where ``None`` means "decide for me".

    ``reasoning`` adds the chain-of-thought allowance, for callers that want the
    whole figure. ``TaskRunner`` passes only ``parser_model`` and adds the
    allowance itself, so it applies whatever the schema looks like; the Studio
    passes only ``reasoning``, to warn when a hand-set value looks too small.
    """
    base = (
        DEFAULT_NUM_PREDICT
        if parser_model is None
        else max(DEFAULT_NUM_PREDICT, estimate_output_tokens(parser_model))
    )
    return base + (REASONING_ALLOWANCE if reasoning else 0)
