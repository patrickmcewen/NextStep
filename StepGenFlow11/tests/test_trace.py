"""Tests for trace capture."""

from src.trace import extract_trace_from_run_result, format_trace_for_analyst


class FakeItem:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_extract_trace_extracts_code():
    items = [
        FakeItem(type="message_output_item", raw_item=FakeItem(content=[
            FakeItem(type="output_text", text="Here is my code:\n```python\ndef build_graph(dims):\n    pass\n```")
        ])),
        FakeItem(type="tool_call_output_item", raw_item=FakeItem(output="Error: no output")),
    ]
    trace = extract_trace_from_run_result(items)
    assert trace["code"] == "def build_graph(dims):\n    pass"
    assert len(trace["tool_outputs"]) == 1
    assert "Error: no output" in trace["tool_outputs"][0]


def test_extract_trace_takes_last_code_block():
    items = [
        FakeItem(type="message_output_item", raw_item=FakeItem(content=[
            FakeItem(type="output_text", text="```python\nfirst_attempt\n```\n\n```python\nsecond_attempt\n```")
        ])),
    ]
    trace = extract_trace_from_run_result(items)
    assert trace["code"] == "second_attempt"


def test_extract_trace_no_code():
    items = [
        FakeItem(type="message_output_item", raw_item=FakeItem(content=[
            FakeItem(type="output_text", text="No code here")
        ])),
    ]
    trace = extract_trace_from_run_result(items)
    assert trace["code"] is None


def test_format_trace_for_analyst():
    traces = [
        {"code": "def build_graph(dims): pass", "tool_outputs": ["Error: something went wrong"]},
        {"code": "def build_graph(dims): return 1", "tool_outputs": ["match=False"]},
    ]
    formatted = format_trace_for_analyst(traces)
    assert "build_graph" in formatted
    assert "Error: something went wrong" in formatted
    assert "Attempt 1" in formatted
    assert "Attempt 2" in formatted
