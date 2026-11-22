import logging

import httpx

from voice_assistant.core.config import Settings

logger = logging.getLogger(__name__)

SEARCH_PATH = "/v1/tools/search"


class DeepSeekSearchClient:
    """Client for the DeepSeek Search API (billed per successful request)."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self._api_key = settings.deepseek_api_key
        self._base_url = settings.deepseek_base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=settings.deepseek_timeout)

    def search(self, query: str, count: int = 5) -> list[dict] | None:
        try:
            response = self._client.post(
                f"{self._base_url}{SEARCH_PATH}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"query": query, "count": count},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("DeepSeek search failed: %s", exc)
            return None

        results = payload.get("results")
        if not isinstance(results, list):
            return []

        return results
