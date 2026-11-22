import logging

from voice_assistant.infrastructure.web.deepseek_search import DeepSeekSearchClient
from voice_assistant.tools.base import Tool

logger = logging.getLogger(__name__)

MAX_RESULTS = 5
MAX_SUMMARY_CHARS = 400

WEB_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Search query. Use the same language as the user's question, "
                "for example 'notizie Italia oggi'."
            ),
        }
    },
    "required": ["query"],
    "additionalProperties": False,
}


def build_web_search_tool(client: DeepSeekSearchClient) -> Tool:
    def web_search(query: str) -> str:
        results = client.search(query, count=MAX_RESULTS)
        if results is None:
            return "Ricerca web non disponibile in questo momento."
        if not results:
            return "Nessun risultato trovato."

        lines = []
        for item in results[:MAX_RESULTS]:
            title = item.get("title") or item.get("url") or "Senza titolo"
            url = item.get("url") or ""
            date = (item.get("date") or "")[:10]
            summary = (item.get("summary") or item.get("snippet") or "").strip()
            if len(summary) > MAX_SUMMARY_CHARS:
                summary = summary[:MAX_SUMMARY_CHARS].rstrip() + "..."

            line = f"- {title}"
            if date:
                line += f" ({date})"
            if url:
                line += f" — {url}"
            if summary:
                line += f"\n  {summary}"
            lines.append(line)

        return "\n".join(lines)

    return Tool(
        name="web_search",
        description=(
            "Search the web for news and current information. Use it when the "
            "answer is not already in the current context, for example news, "
            "sports results or recent events."
        ),
        parameters=WEB_SEARCH_PARAMETERS,
        handler=web_search,
    )
