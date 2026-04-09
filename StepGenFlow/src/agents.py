"""Agent definitions: Writer and Analyst.

Uses plain OpenAI chat completions (no native tool calling) since the vLLM
deployment doesn't have --enable-auto-tool-choice. Tool calls are handled
via text-based <tool_call> tags parsed by the orchestrator.
"""

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel

from src.prompts import build_writer_system_prompt


def make_client(llm_config: dict) -> AsyncOpenAI:
    """Create an AsyncOpenAI client from config dict."""
    return AsyncOpenAI(
        base_url=llm_config["url"],
        api_key=llm_config["api_key"],
        timeout=600,
    )


def make_writer_agent(llm_config: dict) -> Agent:
    """Create the Writer agent (no native tools — orchestrator handles tool execution)."""
    client = make_client(llm_config)
    model = OpenAIChatCompletionsModel(model=llm_config["model"], openai_client=client)
    system_prompt = build_writer_system_prompt()

    return Agent(
        name="StepWriter",
        instructions=system_prompt,
        model=model,
    )


def make_analyst_agent(llm_config: dict) -> Agent:
    """Create the Analyst agent (no tools, pure reasoning)."""
    client = make_client(llm_config)
    model = OpenAIChatCompletionsModel(model=llm_config["model"], openai_client=client)

    return Agent(
        name="StepAnalyst",
        instructions="",  # System prompt is built dynamically per invocation
        model=model,
    )
