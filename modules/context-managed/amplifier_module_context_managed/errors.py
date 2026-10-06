"""Safe, actionable context errors; never include provider payloads or history."""


class CompactionError(RuntimeError):
    def __init__(self, code, detail, *, retryable=False):
        self.code = code
        self.retryable = retryable
        super().__init__(f"Context compaction stopped [{code}]: {detail} Original history is preserved; no foreground request was sent for this step.")
