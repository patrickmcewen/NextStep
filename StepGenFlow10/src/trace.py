"""Trace capture: extract conversation history from Writer agent runs."""

import re


def extract_trace_from_run_result(new_items) -> dict:
    """Extract the last code attempt and all tool outputs from a RunResult's new_items.

    Returns dict with:
        code: str or None — the last Python code block the Writer produced
        tool_outputs: list[str] — all tool call results in order
    """
    code = None
    tool_outputs = []

    for item in new_items:
        item_type = getattr(item, "type", "")

        # Extract code from assistant messages
        if item_type == "message_output_item":
            raw = getattr(item, "raw_item", None)
            if raw:
                content_list = getattr(raw, "content", [])
                for content in content_list:
                    text = getattr(content, "text", "")
                    if text:
                        # Find the last python code block
                        blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
                        if blocks:
                            code = blocks[-1].strip()

        # Extract tool outputs
        if item_type == "tool_call_output_item":
            raw = getattr(item, "raw_item", None)
            output = getattr(raw, "output", None) if raw else None
            if output:
                tool_outputs.append(str(output))

    return {"code": code, "tool_outputs": tool_outputs}


def format_trace_for_analyst(traces: list) -> str:
    """Format a list of trace dicts into a readable string for the Analyst."""
    parts = []
    for i, trace in enumerate(traces):
        parts.append(f"=== Attempt {i + 1} ===")
        if trace.get("code"):
            parts.append(f"Code:\n```python\n{trace['code']}\n```")
        else:
            parts.append("Code: (no code extracted)")
        parts.append("Tool outputs:")
        for j, output in enumerate(trace.get("tool_outputs", [])):
            parts.append(f"  [{j + 1}] {output}")
    return "\n\n".join(parts)
