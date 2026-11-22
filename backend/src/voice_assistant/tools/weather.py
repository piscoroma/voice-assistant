import logging
from time import monotonic

import httpx

from voice_assistant.infrastructure.weather.open_meteo import (
    CACHE_SECONDS,
    DAY_INDEX,
    REQUEST_TIMEOUT_SECONDS,
    SLOT_ORDER,
    current_line,
    day_line,
    fetch_forecast,
    format_slot_line,
    geocode_city,
    slots_by_date,
)
from voice_assistant.tools.base import Tool

logger = logging.getLogger(__name__)

WEATHER_PARAMETERS = {
    "type": "object",
    "properties": {
        "city": {
            "type": "string",
            "description": ("City name. Leave empty to use the user's home city (ASSISTANT_CITY)."),
        },
        "day": {
            "type": "string",
            "enum": list(DAY_INDEX),
            "description": "Day of the forecast, relative to today.",
        },
        "part_of_day": {
            "type": "string",
            "enum": [*SLOT_ORDER, "tutto"],
            "description": ("Part of the day to report, or 'tutto' for the full day."),
        },
    },
    "required": [],
    "additionalProperties": False,
}


def build_weather_tool(
    default_city: str,
    client: httpx.Client | None = None,
) -> Tool:
    http_client = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
    locations: dict[str, tuple[float, float, str] | None] = {}
    cache: dict[str, tuple[float, str]] = {}

    def location_for(city: str) -> tuple[float, float, str] | None:
        key = city.strip().lower()
        if key not in locations:
            locations[key] = geocode_city(http_client, city)
        return locations[key]

    def get_weather(
        city: str = "",
        day: str = "oggi",
        part_of_day: str = "tutto",
    ) -> str:
        target = city.strip() or default_city.strip()
        if not target:
            return "Serve una città per controllare il meteo."

        day_index = DAY_INDEX.get(day.strip().lower(), 0)
        part = part_of_day.strip().lower()
        cache_key = f"{target.lower()}|{day_index}|{part}"

        now = monotonic()
        cached = cache.get(cache_key)
        if cached is not None and now - cached[0] < CACHE_SECONDS:
            return cached[1]

        location = location_for(target)
        if location is None:
            return f"Non ho trovato la città {target}."

        payload = fetch_forecast(http_client, location[0], location[1])
        if payload is None:
            return "Meteo non disponibile in questo momento."

        entry = day_line(payload.get("daily"), day_index)
        if entry is None:
            return "Non ho previsioni per quel giorno."

        date_key, line = entry
        lines = []
        if day_index == 0 and isinstance(payload.get("current"), dict):
            lines.append(current_line(payload["current"], location[2]))
        lines.append(f"Meteo a {location[2]}, {line}.")

        slots = slots_by_date(payload.get("hourly")).get(date_key, {})
        if part in SLOT_ORDER:
            slot_line = format_slot_line(part, slots.get(part))
            lines.append(slot_line or f"  {part}: dato non disponibile.")
        else:
            for slot_name in SLOT_ORDER:
                slot_line = format_slot_line(slot_name, slots.get(slot_name))
                if slot_line:
                    lines.append(slot_line)

        result = "\n".join(lines)
        cache[cache_key] = (now, result)
        return result

    return Tool(
        name="get_weather",
        description=(
            "Get current weather and forecast (today, tomorrow and the day "
            "after, with night/morning/afternoon/evening details) for a city. "
            "Use it for cities other than the user's home city or when the "
            "CONTESTO ATTUALE lacks the requested detail."
        ),
        parameters=WEATHER_PARAMETERS,
        handler=get_weather,
    )
