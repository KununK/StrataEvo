"""Tinyagent public API."""

from .agent import DEFAULT_SYSTEM_PROMPT, Agent
from .model import Model, OpenAICompatibleModel, ScriptedModel
from .tool import Tool, ToolRegistry, tool
from .types import AgentResult, Message, ModelResponse, ToolCall, Usage
from .workspace import Workspace

__all__ = [
    "Agent",
    "AgentResult",
    "DEFAULT_SYSTEM_PROMPT",
    "Message",
    "Model",
    "ModelResponse",
    "OpenAICompatibleModel",
    "ScriptedModel",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "Usage",
    "Workspace",
    "tool",
]
