import json
from types import SimpleNamespace

import httpx

from voice_assistant.infrastructure.web.deepseek_search import DeepSeekSearchClient
from voice_assistant.tools.web_search import build_web_search_tool


def make_settings():
    return SimpleNamespace(
        deepseek_api_key="test-key",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_timeout=5.0,
    )


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=1.0)


def test_web_search_tool_formats_results():
    def handler(request):
        assert request.url.path == "/v1/tools/search"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["query"] == "notizie italia"
        assert payload["count"] == 5
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Titolo notizia",
                        "url": "https://example.com/news",
                        "summary": "Sintesi della notizia.",
                        "date": "2026-11-22T10:00:00+00:00",
                    }
                ]
            },
        )

    client = DeepSeekSearchClient(make_settings(), client=make_client(handler))
    tool = build_web_search_tool(client)

    result = tool.handler(query="notizie italia")

    assert tool.name == "web_search"
    assert "- Titolo notizia (2026-11-22) — https://example.com/news" in result
    assert "Sintesi della notizia." in result


def test_web_search_tool_reports_empty_results():
    def handler(request):
        return httpx.Response(200, json={"results": []})

    tool = build_web_search_tool(DeepSeekSearchClient(make_settings(), client=make_client(handler)))

    assert tool.handler(query="qualcosa") == "Nessun risultato trovato."


def test_web_search_tool_handles_api_failure():
    def handler(request):
        return httpx.Response(500)

    tool = build_web_search_tool(DeepSeekSearchClient(make_settings(), client=make_client(handler)))

    assert "non disponibile" in tool.handler(query="notizie")


def test_web_search_tool_truncates_long_summaries():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Titolo",
                        "url": "https://example.com",
                        "summary": "x" * 600,
                    }
                ]
            },
        )

    tool = build_web_search_tool(DeepSeekSearchClient(make_settings(), client=make_client(handler)))

    result = tool.handler(query="notizie")

    assert "..." in result
    assert "x" * 401 not in result
