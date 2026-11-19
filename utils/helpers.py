import re


def format_sources(sources: list) -> str:
    """Convert source list into readable string for frontend."""
    formatted = []
    for src in sources:
        if src.get("url"):
            formatted.append(f"- {src['title']} ({src['url']})")
        else:
            formatted.append(f"- {src['source']}")
        # Optionally add used strips
        for s in src.get("strips_used", [])[:2]:  # show only first 2
            formatted.append(f"  → {s[:100]}...")
    return "\n".join(formatted)


def parse_relevance_score(content) -> float:
    """Extract a 0-10 score even when the model adds a short label."""
    match = re.search(r"(?<!\d)(10|[0-9])(?:\.\d+)?(?!\d)", str(content or ""))
    if not match:
        return 0.0
    return max(0.0, min(10.0, float(match.group(0))))
