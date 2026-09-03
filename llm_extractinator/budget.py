"""The one place that decides how big the context window is.

``num_ctx`` is a single budget that the prompt and the generated answer share,
and ``num_predict`` is a slice of it. Stating that as an invariant:

    prompt_tokens + num_predict <= num_ctx <= min(hardware_ceiling, model_native_max)

Every context bug in this codebase has been a consequence of computing those two
numbers in different places. The rule used to live in four: ``TaskRunner._context_len``
sized the window, ``PredictionTask.__init__`` raised ``num_predict`` afterwards,
``TaskConfig.__post_init__`` validated the cap against it, and
``DataLoader.adapt_num_predict`` raised it for translation. Each knew part of the
relationship and none knew all of it, which is how a generation budget eight
times the size of its window reached the model without anything objecting.

This module is pure: numbers in, numbers out, no I/O and no configuration
lookups. That is what makes the invariant testable in isolation rather than only
observable at the end of a pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Extra generation budget for a model that reasons before answering. The chain
# of thought is charged to the same num_predict as the answer, and traces of
# 500-3,000 tokens are routine, so a 512-token budget can be spent entirely
# inside <think> with no JSON emitted at all. See model_sizing.md.
REASONING_ALLOWANCE = 5000


class ContextBudgetError(ValueError):
    """The context window cannot hold both the prompt and the generated answer.

    A configuration error rather than a data error: it is fatal for every row,
    not something one row can fail on. Raised naming the term that is wrong, so
    the message says what to change.
    """


@dataclass(frozen=True)
class Budget:
    """A resolved window, and how it was arrived at."""

    num_ctx: int
    num_predict: int
    #: Tokens reserved for the prompt — the longest prompt this run can build,
    #: including its safety margin.
    prompt_tokens: int
    #: What the data actually asked for, before any ceiling was applied.
    fitted_ctx: int
    #: Human-readable origin of ``num_ctx``. One of:
    #: ``"--max_context_len"`` (the user fixed the window explicitly),
    #: ``"fitted to the data"`` (auto, and the prompt is the larger term — the
    #: documents set the size), or ``"fitted to the answer budget"`` (auto, but
    #: ``num_predict`` is the larger term — a generation leash, not a
    #: measurement, set the size). The last is called out because a window that
    #: is mostly answer budget is not "fitted to the data" in any useful sense.
    source: str
    #: The ceiling that bound the window, if one did.
    limited_by: Optional[str] = None
    #: The explicit ``--max_context_len`` this was resolved under, if any. Kept
    #: so a later correction — once the prompt has been measured for real — can
    #: re-resolve under the same constraints instead of guessing at them.
    requested_ctx: Optional[int] = None

    @property
    def prompt_room(self) -> int:
        """Tokens left for the prompt once the answer is reserved."""
        return self.num_ctx - self.num_predict

    @property
    def is_clamped(self) -> bool:
        """True when the window is smaller than the data wanted."""
        return self.num_ctx < self.fitted_ctx

    def describe(self) -> str:
        """One line, for the log: where the window went.

        The point is that someone who does not think in tokens can still see
        that the answer and the prompt are competing for the same space. When
        ``num_predict`` is the larger of the two, the line says so outright:
        that is the case where the window size came from a generation leash
        rather than from the documents, and calling it "fitted to the data"
        would hide exactly the thing worth noticing.
        """
        headroom = self.num_ctx - self.prompt_tokens - self.num_predict
        origin = self.source
        if self.limited_by:
            origin += f", capped by {self.limited_by}"
        if self.num_predict > self.prompt_tokens:
            origin += "; answer budget is the larger term"
        return (
            f"context {self.num_ctx} = {self.prompt_tokens} prompt "
            f"+ {self.num_predict} answer "
            f"+ {max(headroom, 0)} headroom ({origin})"
        )


def _tightest_ceiling(
    hardware_ceiling: Optional[int], model_native_max: Optional[int]
) -> tuple[Optional[int], Optional[str]]:
    """Whichever ceiling binds first, and its name for the message."""
    candidates = [
        (hardware_ceiling, "--max_context_cap"),
        (model_native_max, "the model's native context length"),
    ]
    present = [(value, name) for value, name in candidates if value is not None]
    if not present:
        return None, None
    return min(present, key=lambda pair: pair[0])


def resolve_budget(
    *,
    prompt_tokens: int,
    output_tokens: int,
    requested_ctx: Optional[int] = None,
    hardware_ceiling: Optional[int] = None,
    model_native_max: Optional[int] = None,
) -> Budget:
    """Resolve the window and the generation budget together.

    ``prompt_tokens``
        The longest prompt this run can build, safety margin included.
    ``output_tokens``
        What generation needs — the reasoning allowance already added if the
        model thinks. This is an input, not something decided here: deciding it
        after the window was sized is the bug this module exists to prevent.
    ``requested_ctx``
        An explicit ``--max_context_len N``. ``None`` means fit to the data.
    ``hardware_ceiling``
        ``--max_context_cap``: what the GPU can afford. Ollama allocates the KV
        cache up front, so a generous window costs memory whether it is used or
        not.
    ``model_native_max``
        What the model itself supports, from ``/api/show``. Asking for more than
        this does not give more; it either errors or degrades.

    A ceiling below what the data wants is *clamped*, not rejected: capping the
    window so a handful of outlier documents get truncated is a legitimate trade
    when the alternative is not running at all. The caller is expected to say how
    many documents that affects. A ceiling that cannot even hold the answer is a
    different thing entirely, and raises.
    """
    if output_tokens <= 0:
        raise ContextBudgetError(
            f"The generation budget must be positive, got num_predict={output_tokens}."
        )
    if prompt_tokens < 0:
        raise ContextBudgetError(
            f"Prompt tokens cannot be negative, got {prompt_tokens}."
        )

    fitted = prompt_tokens + output_tokens
    if requested_ctx is not None:
        if requested_ctx <= 0:
            raise ContextBudgetError(
                f"max_context_len must be positive, got {requested_ctx}."
            )
        num_ctx, source = requested_ctx, "--max_context_len"
    elif output_tokens > prompt_tokens:
        # The window is mostly generation budget: num_predict, not the
        # documents, set its size. num_predict is a leash the caller chose, not
        # a measurement, so "fitted to the data" would misdescribe where the
        # number came from. See PLAN_0.8.0 C1 and .claude/rules/context-budget.md.
        num_ctx, source = fitted, "fitted to the answer budget"
    else:
        num_ctx, source = fitted, "fitted to the data"

    ceiling, ceiling_name = _tightest_ceiling(hardware_ceiling, model_native_max)
    limited_by = None
    if ceiling is not None and num_ctx > ceiling:
        num_ctx, limited_by = ceiling, ceiling_name

    if output_tokens >= num_ctx:
        blame = limited_by or source
        raise ContextBudgetError(
            f"num_predict ({output_tokens}) must be smaller than num_ctx "
            f"({num_ctx}): the context window has to hold the prompt as well as "
            f"the generated answer, and this leaves "
            f"{max(num_ctx - output_tokens, 0)} tokens for it. The window is "
            f"{num_ctx} because of {blame}. Either raise it or lower "
            f"--num_predict."
        )

    return Budget(
        num_ctx=num_ctx,
        num_predict=output_tokens,
        prompt_tokens=min(prompt_tokens, num_ctx - output_tokens),
        fitted_ctx=fitted,
        source=source,
        limited_by=limited_by,
        requested_ctx=requested_ctx,
    )
