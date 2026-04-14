"""Exception hierarchy for CogniGraph."""


class CogniGraphError(Exception):
    """Base exception for all CogniGraph errors."""


class NodeNotFoundError(CogniGraphError):
    """Raised when a graph node lookup fails."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        super().__init__(f"Node not found: {node_id}")


class EmbeddingError(CogniGraphError):
    """Raised when the embedding model fails."""


class LLMError(CogniGraphError):
    """Raised when the LLM API call fails."""


class LLMRetriableError(LLMError):
    """LLM failure that the caller may safely retry (rate limit, timeout, transient network)."""


class LLMPermanentError(LLMError):
    """LLM failure that will not succeed on retry (auth, bad request, permission denied)."""


class SafetyViolationError(CogniGraphError):
    """Raised when the safety boundary is triggered."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Safety violation: {reason}")


class CapacityExceededError(CogniGraphError):
    """Raised when the graph is at maximum capacity."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        super().__init__(f"Graph capacity exceeded: {capacity} nodes")


class PersistenceError(CogniGraphError):
    """Raised when SQLite read/write fails."""
