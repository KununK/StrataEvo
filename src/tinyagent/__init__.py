"""Tinyagent public API."""

from .agent import DEFAULT_SYSTEM_PROMPT, Agent
from .events import Event, EventBus
from .model import Model, OpenAICompatibleModel, ScriptedModel
from .session import SessionStore
from .tool import Tool, ToolRegistry, allow_all, terminal_approval, tool
from .types import AgentResult, Message, ModelResponse, ToolCall, Usage
from .workspace import Workspace

__all__ = [
    "Agent",
    "AgentResult",
    "DEFAULT_SYSTEM_PROMPT",
    "Event",
    "EventBus",
    "Message",
    "Model",
    "ModelResponse",
    "OpenAICompatibleModel",
    "ScriptedModel",
    "SessionStore",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "Usage",
    "Workspace",
    "allow_all",
    "terminal_approval",
    "tool",
]
