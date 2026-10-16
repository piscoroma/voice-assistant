import logging

from openai import OpenAI

from voice_assistant.core.config import Settings

logger = logging.getLogger(__name__)


class DeepSeekProvider:
    def __init__(self, settings: Settings) -> None:
        self._model = settings.deepseek_model
        self._client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=settings.deepseek_timeout,
            max_retries=settings.deepseek_max_retries,
        )

    def generate_response(
        self,
        history: list[dict[str, str]],
        user_message: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Sei un assistente vocale domestico. "
                    "Rispondi in italiano in modo naturale, "
                    "chiaro e conciso. Evita Markdown e liste lunghe."
                ),
            },
            *history,
            {"role": "user", "content": user_message},
        ]

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=300,
            extra_body={"thinking": {"type": "disabled"}},
        )

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            raise RuntimeError("DeepSeek returned an empty response")

        return answer.strip()
