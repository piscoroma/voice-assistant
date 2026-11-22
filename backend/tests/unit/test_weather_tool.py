import httpx

from voice_assistant.tools.weather import build_weather_tool


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=1.0)


def open_meteo_handler(calls, city_name="Torino", has_results=True):
    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "geocoding-api.open-meteo.com":
            if not has_results:
                return httpx.Response(200, json={"results": []})
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "latitude": 45.07,
                            "longitude": 7.68,
                            "name": city_name,
                        }
                    ]
                },
            )

        return httpx.Response(
            200,
            json={
                "current": {
                    "time": "2026-11-22T15:30",
                    "temperature_2m": 12.3,
                    "apparent_temperature": 11.0,
                    "relative_humidity_2m": 70,
                    "weather_code": 0,
                    "wind_speed_10m": 5.0,
                },
                "daily": {
                    "time": ["2026-11-22", "2026-11-23", "2026-11-24"],
                    "weather_code": [1, 61, 3],
                    "temperature_2m_max": [10.4, 9.2, 8.0],
                    "temperature_2m_min": [3.1, 2.4, 1.0],
                    "precipitation_probability_max": [10, 60, 20],
                },
                "hourly": {
                    "time": [
                        "2026-11-22T03:00",
                        "2026-11-22T09:00",
                        "2026-11-22T15:00",
                        "2026-11-22T21:00",
                        "2026-11-23T03:00",
                        "2026-11-23T09:00",
                        "2026-11-23T15:00",
                        "2026-11-23T21:00",
                        "2026-11-24T03:00",
                        "2026-11-24T09:00",
                        "2026-11-24T15:00",
                        "2026-11-24T21:00",
                    ],
                    "temperature_2m": [
                        4.0,
                        5.0,
                        8.0,
                        6.0,
                        4.0,
                        6.0,
                        13.0,
                        8.0,
                        4.0,
                        5.0,
                        9.0,
                        7.0,
                    ],
                    "weather_code": [61, 61, 61, 61, 45, 45, 1, 3, 61, 61, 61, 61],
                    "precipitation_probability": [
                        100,
                        100,
                        100,
                        100,
                        0,
                        0,
                        10,
                        10,
                        70,
                        70,
                        70,
                        70,
                    ],
                },
            },
        )

    return handler


def test_weather_tool_uses_default_city_and_day_part():
    calls = []
    tool = build_weather_tool("Torino", client=make_client(open_meteo_handler(calls)))

    result = tool.handler(day="domani", part_of_day="pomeriggio")

    assert tool.name == "get_weather"
    assert "Meteo a Torino, lunedì 23/11: 2/9 °C, pioggia debole" in result
    assert "pomeriggio: 13 °C, prevalentemente sereno, precipitazioni 10%." in result
    assert "notte:" not in result
    assert "Meteo attuale" not in result


def test_weather_tool_reports_current_conditions_for_today():
    calls = []
    tool = build_weather_tool("Torino", client=make_client(open_meteo_handler(calls)))

    result = tool.handler()

    assert "Meteo attuale a Torino: 12.3 °C" in result
    assert "notte:" in result
    assert "mattina:" in result
    assert "pomeriggio:" in result
    assert "sera:" in result


def test_weather_tool_accepts_another_city():
    seen_cities = []

    def handler(request):
        if request.url.host == "geocoding-api.open-meteo.com":
            seen_cities.append(request.url.params["name"])
            return httpx.Response(
                200,
                json={"results": [{"latitude": 45.46, "longitude": 9.19, "name": "Milano"}]},
            )
        return open_meteo_handler([], city_name="Milano")(request)

    tool = build_weather_tool("Torino", client=make_client(handler))

    result = tool.handler(city="Milano", day="domani")

    assert seen_cities == ["Milano"]
    assert "Meteo a Milano" in result


def test_weather_tool_caches_repeated_requests():
    calls = []
    tool = build_weather_tool("Torino", client=make_client(open_meteo_handler(calls)))

    first = tool.handler(day="domani", part_of_day="pomeriggio")
    second = tool.handler(day="domani", part_of_day="pomeriggio")

    assert first == second
    assert calls == ["geocoding-api.open-meteo.com", "api.open-meteo.com"]


def test_weather_tool_handles_unknown_city():
    calls = []
    tool = build_weather_tool(
        "Torino",
        client=make_client(open_meteo_handler(calls, has_results=False)),
    )

    assert "Non ho trovato la città Torino" in tool.handler()


def test_weather_tool_handles_api_failure():
    def handler(request):
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(
                200,
                json={"results": [{"latitude": 45.07, "longitude": 7.68, "name": "Torino"}]},
            )
        return httpx.Response(500)

    tool = build_weather_tool("Torino", client=make_client(handler))

    assert "non disponibile" in tool.handler()


def test_weather_tool_without_any_city():
    def handler(request):
        raise AssertionError("no HTTP request expected without a city")

    tool = build_weather_tool("", client=make_client(handler))

    assert "Serve una città" in tool.handler()
