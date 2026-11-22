from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Tool:
    """A callable exposed to the LLM through function calling."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
