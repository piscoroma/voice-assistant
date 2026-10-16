from voice_assistant.infrastructure.llm.deepseek import DeepSeekProvider


class LLMService:
    def __init__(self, provider: DeepSeekProvider) -> None:
        self._provider = provider

    def generate_response(
        self,
        history: list[dict[str, str]],
        user_message: str,
    ) -> str:
        return self._provider.generate_response(
            history=history,
            user_message=user_message,
        )
