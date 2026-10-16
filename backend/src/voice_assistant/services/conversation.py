import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Session:
    history: deque[tuple[str, str]] = field(default_factory=deque)
    last_activity: float = 0.0


class ConversationService:
    def __init__(
        self,
        max_turns: int = 5,
        ttl_seconds: int = 600,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")

        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be at least 1")

        self._max_turns = max_turns
        self._ttl_seconds = ttl_seconds
        self._clock = clock or time.monotonic

        self._sessions: dict[str, Session] = {}

    def add_exchange(
        self,
        esp_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        self._validate_esp_id(esp_id)

        if not user_message.strip():
            raise ValueError("user_message cannot be empty")

        if not assistant_message.strip():
            raise ValueError("assistant_message cannot be empty")

        now = self._clock()
        self._remove_expired(now)

        session = self._sessions.get(esp_id)

        if session is None:
            session = Session(history=deque(maxlen=self._max_turns))
            self._sessions[esp_id] = session

        session.history.append((user_message, assistant_message))
        session.last_activity = now

    def get_history(
        self,
        esp_id: str,
    ) -> list[dict[str, str]]:
        self._validate_esp_id(esp_id)

        now = self._clock()
        self._remove_expired(now)

        session = self._sessions.get(esp_id)

        if session is None:
            return []

        messages: list[dict[str, str]] = []

        for user_message, assistant_message in session.history:
            messages.append(
                {
                    "role": "user",
                    "content": user_message,
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message,
                }
            )

        return messages

    def cleanup_expired(self) -> int:
        """Remove expired sessions and return the number removed."""
        return self._remove_expired(self._clock())

    def _remove_expired(self, now: float) -> int:
        expired_ids = [
            esp_id
            for esp_id, session in self._sessions.items()
            if now - session.last_activity >= self._ttl_seconds
        ]

        for esp_id in expired_ids:
            del self._sessions[esp_id]

        return len(expired_ids)

    @staticmethod
    def _validate_esp_id(esp_id: str) -> None:
        if not esp_id or not esp_id.strip():
            raise ValueError("esp_id cannot be empty")
