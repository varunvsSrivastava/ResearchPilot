from dataclasses import dataclass

@dataclass(frozen=True)
class TokenUsage:
    """Token consumption for a single LLM request."""

    input_tokens: int
    output_tokens: int
    total_tokens: int

@dataclass
class UsageTracker:
    """Track cumulative token consumption across LLM requests."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    def record(self, usage: TokenUsage) -> None:
        """Record toekn usage for a completed request."""

        self.total_requests += 1
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens

    def get_summary(self) -> dict[str, int]:
        """Return cumulative token usage."""

        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens
        }

    def reset(self) -> None:
        """Reset cumulative usage statistics."""

        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0

usage_tracker = UsageTracker()

def track_usage(response) -> TokenUsage | None:
    """Extract and record token usage from an API response."""

    usage = getattr(response, "usage", None)

    if usage is None:
        return None

    token_usage = TokenUsage(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        total_tokens=usage.total_tokens or 0,
    )

    usage_tracker.record(token_usage)

    return token_usage

