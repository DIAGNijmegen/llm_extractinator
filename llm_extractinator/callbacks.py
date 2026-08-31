from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from tqdm.auto import tqdm


class BatchCallBack(BaseCallbackHandler):
    """Progress across *rows*, not across calls to the model.

    ``on_llm_end`` fires once per LLM call, and one row can make more than one —
    a retried transport error, for instance. Ticking on every call meant a bar
    built for 5 rows reporting ``15it``, which reads as "everything finished"
    immediately before the output turns out to be empty. That is the visible
    half of the "it says it passed and then the rows are blank" complaint.

    Rows are identified by ``parent_run_id``: every attempt for a row shares the
    chain run that invoked it, so counting distinct parents counts rows.
    ``run_id`` is the fallback for a bare model invocation with no parent.

    ``on_llm_error`` is handled too. Without it a row whose call failed never
    ticked at all, so the bar simply stopped short of its total with no
    indication why.
    """

    def __init__(self, total: int):
        super().__init__()
        self.total = total
        self.llm_errors = 0
        self._rows_seen: set = set()
        self.progress_bar = tqdm(total=total, ascii=True, dynamic_ncols=False)

    @property
    def count(self) -> int:
        """Rows finished. Was a public attribute; kept as a derived value."""
        return len(self._rows_seen)

    def _row_finished(self, run_id: UUID, parent_run_id: Optional[UUID]) -> None:
        key = parent_run_id or run_id
        if key in self._rows_seen:
            return
        self._rows_seen.add(key)
        self.progress_bar.update(1)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self._row_finished(run_id, parent_run_id)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        # Counts erroring *calls*, not failed rows: an error followed by a
        # successful retry is counted here and still ends as a good row. The
        # authoritative failure count comes from Predictor._log_outcome.
        self.llm_errors += 1
        self.progress_bar.set_postfix(errors=self.llm_errors)
        self._row_finished(run_id, parent_run_id)
