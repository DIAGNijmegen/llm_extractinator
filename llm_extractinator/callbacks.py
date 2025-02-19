from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from tqdm.auto import tqdm


class BatchCallBack(BaseCallbackHandler):
    """A callback handler to manage progress during batch processing."""

    def __init__(self, total: int):
        super().__init__()
        self.count = 0
        self.progress_bar = tqdm(total=total)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self.count += 1
        self.progress_bar.update(1)
