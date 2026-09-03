"""Helpers for Reddit-side JSONL (Pushshift-style and ESRC-Micro compact keys)."""


def comment_body(obj: dict) -> str:
    """Return comment text from either long-form (`body`) or micro-dataset (`b`)."""
    b = obj.get("body")
    if isinstance(b, str) and b:
        return b
    short = obj.get("b")
    return short if isinstance(short, str) else ""
