"""
Abstract base for all Cyrus LangChain agents.
Handles LLM init, tool binding, and structured output parsing.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from core.config import settings

log = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Wraps a LangChain ChatOpenAI instance with:
    - System prompt definition (subclass provides)
    - Optional tool binding
    - Structured JSON output via prompt contract
    - Action logging hook
    """

    name: str  # agent identifier, set by subclass

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        llm_kwargs: dict[str, Any] = {
            "model": settings.LLM_MODEL,
            "temperature": settings.LLM_TEMPERATURE,
            "api_key": settings.FIREWORKS_KEY,
            "base_url": settings.FIREWORKS_URL,
        }

        self._llm = ChatOpenAI(**llm_kwargs)

        if tools:
            self._llm = self._llm.bind_tools(tools)
            self._tools = {t.name: t for t in tools}
        else:
            self._tools = {}

        log.info(
            "Agent %s initialised (model=%s, tools=%s)",
            self.name,
            settings.LLM_MODEL,
            list(self._tools.keys()),
        )

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Static system prompt defining this agent's role and output format."""
        ...

    def invoke(self, user_message: str) -> str:
        """Single-turn invocation. Returns the LLM text response."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message),
        ]

        response = self._llm.invoke(messages)

        # Handle tool calls if the LLM requested them
        if hasattr(response, "tool_calls") and response.tool_calls:
            return self._handle_tool_calls(response, messages)

        return response.content

    def _handle_tool_calls(self, response, messages: list) -> str:
        """
        Execute tool calls requested by the LLM and continue the conversation.
        Runs up to 5 rounds before forcing a final text response.
        """
        from langchain_core.messages import AIMessage, ToolMessage

        messages = list(messages)
        messages.append(response)

        for _ in range(5):
            tool_results = []
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                if tool_name in self._tools:
                    try:
                        result = self._tools[tool_name].invoke(tool_args)
                        log.info("[%s] Tool %s → %s", self.name, tool_name, str(result)[:200])
                    except Exception as exc:
                        result = f"ERROR: {exc}"
                        log.error("[%s] Tool %s failed: %s", self.name, tool_name, exc)
                else:
                    result = f"Tool {tool_name!r} not found"

                tool_results.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

            messages.extend(tool_results)
            response = self._llm.invoke(messages)
            messages.append(response)

            if not (hasattr(response, "tool_calls") and response.tool_calls):
                break

        return response.content