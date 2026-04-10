"""Agent definitions for StepGenFlow4.

Creates agents for lowering passes, translator passes, the Writer, and the Analyst.
Uses plain OpenAI chat completions (no native tool calling) since the vLLM
deployment doesn't have --enable-auto-tool-choice.
"""

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel

from src.prompts import build_pass_system_prompt, build_writer_system_prompt, build_judge_system_prompt


def make_client(llm_config: dict) -> AsyncOpenAI:
    """Create an AsyncOpenAI client from config dict."""
    return AsyncOpenAI(
        base_url=llm_config["url"],
        api_key=llm_config["api_key"],
        timeout=600,
    )


def make_pass_agent(llm_config: dict, pass_name: str) -> Agent:
    """Create an agent for any pass (lowering or translator)."""
    client = make_client(llm_config)
    model = OpenAIChatCompletionsModel(model=llm_config["model"], openai_client=client)
    system_prompt = build_pass_system_prompt(pass_name)

    return Agent(
        name=f"StepPass_{pass_name}",
        instructions=system_prompt,
        model=model,
    )


def make_writer_agent(llm_config: dict) -> Agent:
    """Create the Writer agent (fallback when translation pipeline fails)."""
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


def make_judge_agent(llm_config: dict, pass_name: str) -> Agent:
    """Create a judge agent for format compliance checking."""
    client = make_client(llm_config)
    model = OpenAIChatCompletionsModel(model=llm_config["model"], openai_client=client)
    system_prompt = build_judge_system_prompt(pass_name)

    return Agent(
        name=f"StepJudge_{pass_name}",
        instructions=system_prompt,
        model=model,
    )


