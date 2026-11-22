import httpx

from voice_assistant.infrastructure.weather.open_meteo import OpenMeteoContext


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=1.0)


def open_meteo_handler(calls, include_forecast=True):
    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "latitude": 45.07,
                            "longitude": 7.68,
                            "name": "Torino",
                        }
                    ]
                },
            )

        payload = {
            "current": {
                "time": "2026-11-22T15:30",
                "temperature_2m": 12.3,
                "apparent_temperature": 11.0,
                "relative_humidity_2m": 70,
                "weather_code": 0,
                "wind_speed_10m": 5.0,
            }
        }
        if include_forecast:
            payload["daily"] = {
                "time": ["2026-11-22", "2026-11-23", "2026-11-24"],
                "weather_code": [1, 61, 3],
                "temperature_2m_max": [10.4, 9.2, 8.0],
                "temperature_2m_min": [3.1, 2.4, 1.0],
                "precipitation_probability_max": [10, 60, 20],
            }
            payload["hourly"] = {
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
                "temperature_2m": [4.0, 5.0, 8.0, 6.0, 4.0, 6.0, 13.0, 8.0, 4.0, 5.0, 9.0, 7.0],
                "weather_code": [61, 61, 61, 61, 45, 45, 1, 3, 61, 61, 61, 61],
                "precipitation_probability": [100, 100, 100, 100, 0, 0, 10, 10, 70, 70, 70, 70],
            }

        return httpx.Response(200, json=payload)

    return handler


def test_current_context_includes_time_and_weather():
    calls = []
    context = OpenMeteoContext("Torino", client=make_client(open_meteo_handler(calls)))

    result = context.current_context()

    assert "domenica 22/11/2026 15:30" in result
    assert "Meteo attuale a Torino: 12.3 °C" in result
    assert "umidità 70%" in result
    assert "vento 5 km/h" in result
    assert "cielo sereno" in result
    assert "Previsioni a Torino:" in result
    assert "- domenica 22/11: 3/10 °C, prevalentemente sereno" in result
    assert "- lunedì 23/11: 2/9 °C, pioggia debole" in result
    assert "probabilità di precipitazioni 60%" in result
    assert "  pomeriggio: 13 °C, prevalentemente sereno, precipitazioni 10%." in result
    assert "  notte: 4 °C, nebbia, precipitazioni 0%." in result
    assert "  sera: 7 °C, pioggia debole, precipitazioni 70%." in result
    assert calls == ["geocoding-api.open-meteo.com", "api.open-meteo.com"]


def test_current_context_without_daily_forecast():
    calls = []
    context = OpenMeteoContext(
        "Torino",
        client=make_client(open_meteo_handler(calls, include_forecast=False)),
    )

    result = context.current_context()

    assert "Meteo attuale a Torino" in result
    assert "Previsioni a" not in result


def test_current_context_is_cached():
    calls = []
    context = OpenMeteoContext("Torino", client=make_client(open_meteo_handler(calls)))

    first = context.current_context()
    second = context.current_context()

    assert first == second
    assert len(calls) == 2


def test_current_context_falls_back_to_time_only_on_failure():
    def handler(request):
        return httpx.Response(500)

    context = OpenMeteoContext("Torino", client=make_client(handler))

    result = context.current_context()

    assert "Ora e data locali" in result
    assert "Meteo a" not in result


def test_current_context_without_city_makes_no_requests():
    def handler(request):
        raise AssertionError("no HTTP request expected without a city")

    context = OpenMeteoContext("   ", client=make_client(handler))

    result = context.current_context()

    assert "Ora e data locali" in result
    assert "Meteo a" not in result
