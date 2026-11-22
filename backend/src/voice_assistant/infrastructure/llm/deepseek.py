import json
import logging
from collections.abc import Callable
from time import perf_counter

from openai import OpenAI

from voice_assistant.core.config import Settings
from voice_assistant.tools.base import Tool

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 3


class DeepSeekProvider:
    def __init__(
        self,
        settings: Settings,
        context_provider: Callable[[], str] | None = None,
        tools: list[Tool] | None = None,
    ) -> None:
        self._model = settings.deepseek_model
        self._tools = tools or []
        self._tools_by_name = {tool.name: tool for tool in self._tools}
        self._system_prompt = self._build_system_prompt(
            settings.assistant_city,
            has_web_search="web_search" in self._tools_by_name,
            has_weather_tool="get_weather" in self._tools_by_name,
        )
        self._context_provider = context_provider
        self._client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=settings.deepseek_timeout,
            max_retries=settings.deepseek_max_retries,
        )

    @staticmethod
    def _build_system_prompt(
        city: str,
        has_web_search: bool,
        has_weather_tool: bool,
    ) -> str:
        prompt = (
            "Sei un assistente vocale domestico. "
            "Rispondi in italiano in modo naturale, "
            "chiaro e conciso. Evita Markdown e liste lunghe."
        )
        city = city.strip()
        if city:
            prompt += f" L'utente vive a {city}: tienilo presente come contesto."
        prompt += (
            " Quando è presente un blocco CONTESTO ATTUALE, usalo per rispondere"
            " a domande su data, ora, meteo attuale e previsioni dei prossimi"
            " giorni, comprese le fasce mattina, pomeriggio, sera e notte, in"
            " modo diretto. Non inventare dati che non hai."
        )
        if has_weather_tool:
            prompt += (
                " Hai il tool get_weather per altre città o per dettagli non"
                " presenti nel CONTESTO ATTUALE: in quei casi usalo invece di"
                " inventare."
            )
        if has_web_search:
            prompt += (
                " Hai il tool web_search per notizie e informazioni attuali:"
                " usalo quando la risposta non è nel CONTESTO ATTUALE, cita"
                " brevemente la fonte e non menzionare i tool all'utente."
            )
        return prompt

    def _system_content(self) -> str:
        if self._context_provider is None:
            return self._system_prompt

        context = self._context_provider().strip()
        if not context:
            return self._system_prompt

        return f"{self._system_prompt}\n\nCONTESTO ATTUALE\n{context}"

    def generate_response(
        self,
        history: list[dict[str, str]],
        user_message: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": self._system_content(),
            },
            *history,
            {"role": "user", "content": user_message},
        ]

        for _ in range(MAX_TOOL_ROUNDS + 1):
            message = self._chat(messages)
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                return self._extract_answer(message)

            messages.append(self._assistant_tool_message(message, tool_calls))
            messages.extend(self._run_tool(call) for call in tool_calls)

        raise RuntimeError("Tool call limit exceeded")

    def _chat(self, messages: list[dict]) -> object:
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 300,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if self._tools:
            kwargs["tools"] = [tool.schema for tool in self._tools]

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message

    @staticmethod
    def _assistant_tool_message(message: object, tool_calls: list) -> dict:
        return {
            "role": "assistant",
            "content": getattr(message, "content", None),
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ],
        }

    def _run_tool(self, call: object) -> dict:
        name = call.function.name
        tool = self._tools_by_name.get(name)

        if tool is None:
            logger.warning("Model requested unknown tool: %s", name)
            content = f"Errore: il tool {name} non esiste."
        else:
            try:
                arguments = json.loads(call.function.arguments or "{}")
                logger.info(
                    "Tool %s called with %s",
                    name,
                    json.dumps(arguments, ensure_ascii=False)[:500],
                )
                start = perf_counter()
                content = tool.handler(**arguments)
                logger.info(
                    "Tool %s completed in %.3f seconds",
                    name,
                    perf_counter() - start,
                )
            except Exception as exc:  # noqa: BLE001 - a tool failure must not break the turn
                logger.warning("Tool %s failed: %s", name, exc)
                content = f"Errore nell'esecuzione del tool {name}."

        return {
            "role": "tool",
            "tool_call_id": call.id,
            "content": content,
        }

    @staticmethod
    def _extract_answer(message: object) -> str:
        answer = getattr(message, "content", None)

        if not answer or not answer.strip():
            raise RuntimeError("DeepSeek returned an empty response")

        return answer.strip()
