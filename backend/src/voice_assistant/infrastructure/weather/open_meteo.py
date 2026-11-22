import logging
import time
from collections import Counter
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 5.0
CACHE_SECONDS = 600

WEEKDAYS = (
    "lunedì",
    "martedì",
    "mercoledì",
    "giovedì",
    "venerdì",
    "sabato",
    "domenica",
)

WEATHER_DESCRIPTIONS = {
    0: "cielo sereno",
    1: "prevalentemente sereno",
    2: "parzialmente nuvoloso",
    3: "coperto",
    45: "nebbia",
    48: "nebbia con brina",
    51: "pioviggine debole",
    53: "pioviggine moderata",
    55: "pioviggine intensa",
    56: "pioviggine gelata debole",
    57: "pioviggine gelata intensa",
    61: "pioggia debole",
    63: "pioggia moderata",
    65: "pioggia intensa",
    66: "pioggia gelata debole",
    67: "pioggia gelata intensa",
    71: "neve debole",
    73: "neve moderata",
    75: "neve intensa",
    77: "granelli di neve",
    80: "rovesci deboli",
    81: "rovesci moderati",
    82: "rovesci intensi",
    85: "rovesci di neve deboli",
    86: "rovesci di neve intensi",
    95: "temporale",
    96: "temporale con grandine debole",
    99: "temporale con grandine forte",
}
DEFAULT_DESCRIPTION = "condizioni non disponibili"

SLOT_ORDER = ("notte", "mattina", "pomeriggio", "sera")
DAY_INDEX = {"oggi": 0, "domani": 1, "dopodomani": 2}


def describe_code(code: int | None) -> str:
    return WEATHER_DESCRIPTIONS.get(code, DEFAULT_DESCRIPTION)


def slot_for_hour(hour: int) -> str | None:
    if 0 <= hour < 6:
        return "notte"
    if 6 <= hour < 14:
        return "mattina"
    if 14 <= hour < 19:
        return "pomeriggio"
    if 19 <= hour < 24:
        return "sera"
    return None


def geocode_city(
    client: httpx.Client,
    city: str,
) -> tuple[float, float, str] | None:
    try:
        response = client.get(
            GEOCODING_URL,
            params={
                "name": city,
                "count": 1,
                "language": "it",
                "format": "json",
            },
        )
        response.raise_for_status()
        results = response.json().get("results") or []
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        logger.warning("Open-Meteo geocoding failed: %s", exc)
        return None

    if not results:
        logger.warning("Open-Meteo geocoding found no match for %r", city)
        return None

    first = results[0]
    try:
        return (
            float(first["latitude"]),
            float(first["longitude"]),
            first.get("name") or city,
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Open-Meteo geocoding returned invalid data: %s", exc)
        return None


def fetch_forecast(
    client: httpx.Client,
    latitude: float,
    longitude: float,
    forecast_days: int = 3,
) -> dict | None:
    try:
        response = client.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,apparent_temperature,"
                    "relative_humidity_2m,weather_code,wind_speed_10m"
                ),
                "daily": (
                    "weather_code,temperature_2m_max,"
                    "temperature_2m_min,precipitation_probability_max"
                ),
                "hourly": "temperature_2m,weather_code,precipitation_probability",
                "forecast_days": forecast_days,
                "timezone": "auto",
            },
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Open-Meteo weather request failed: %s", exc)
        return None

    if not isinstance(payload, dict) or "current" not in payload:
        logger.warning("Open-Meteo returned an unexpected payload")
        return None

    return payload


def current_line(current: dict, name: str) -> str:
    return (
        f"Meteo attuale a {name}: {current['temperature_2m']:.1f} °C "
        f"(percepiti {current['apparent_temperature']:.1f} °C), "
        f"umidità {current['relative_humidity_2m']:.0f}%, "
        f"vento {current['wind_speed_10m']:.0f} km/h, "
        f"{describe_code(current['weather_code'])}."
    )


def day_line(daily: dict | None, index: int) -> tuple[str, str] | None:
    if not daily:
        return None

    dates = daily.get("time") or []
    if index >= len(dates):
        return None

    codes = daily.get("weather_code") or []
    maxima = daily.get("temperature_2m_max") or []
    minima = daily.get("temperature_2m_min") or []
    rains = daily.get("precipitation_probability_max") or []

    try:
        moment = datetime.fromisoformat(dates[index])
        code = codes[index]
        high = maxima[index]
        low = minima[index]
    except (IndexError, TypeError, ValueError):
        return None

    if high is None or low is None:
        return None

    line = (
        f"{WEEKDAYS[moment.weekday()]} {moment.strftime('%d/%m')}: "
        f"{low:.0f}/{high:.0f} °C, {describe_code(code)}"
    )
    rain = rains[index] if index < len(rains) else None
    if rain is not None:
        line += f", probabilità di precipitazioni {rain:.0f}%"

    return dates[index], line


def slots_by_date(hourly: dict | None) -> dict[str, dict[str, dict[str, list]]]:
    if not hourly:
        return {}

    times = hourly.get("time") or []
    temperatures = hourly.get("temperature_2m") or []
    codes = hourly.get("weather_code") or []
    rains = hourly.get("precipitation_probability") or []

    slots: dict[str, dict[str, dict[str, list]]] = {}
    for index, time_value in enumerate(times):
        try:
            moment = datetime.fromisoformat(time_value)
            temperature = temperatures[index]
        except (IndexError, TypeError, ValueError):
            continue

        if temperature is None:
            continue

        slot_name = slot_for_hour(moment.hour)
        if slot_name is None:
            continue

        bucket = slots.setdefault(moment.date().isoformat(), {}).setdefault(
            slot_name, {"temps": [], "codes": [], "rains": []}
        )
        bucket["temps"].append(float(temperature))

        code = codes[index] if index < len(codes) else None
        if code is not None:
            bucket["codes"].append(int(code))

        rain = rains[index] if index < len(rains) else None
        if rain is not None:
            bucket["rains"].append(float(rain))

    return slots


def format_slot_line(slot_name: str, slot: dict[str, list] | None) -> str:
    if not slot or not slot["temps"]:
        return ""

    low = min(slot["temps"])
    high = max(slot["temps"])
    if round(low) == round(high):
        temperature = f"{low:.0f} °C"
    else:
        temperature = f"{low:.0f}-{high:.0f} °C"

    parts = [temperature]
    if slot["codes"]:
        code = Counter(slot["codes"]).most_common(1)[0][0]
        parts.append(describe_code(code))
    if slot["rains"]:
        parts.append(f"precipitazioni {max(slot['rains']):.0f}%")

    return f"  {slot_name}: " + ", ".join(parts) + "."


class OpenMeteoContext:
    """Current date/time and weather for the configured city.

    Uses Open-Meteo (no API key). Network failures degrade to a time-only
    context so the conversation still works offline.
    """

    def __init__(self, city: str, client: httpx.Client | None = None) -> None:
        self._city = city.strip()
        self._client = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
        self._location: tuple[float, float, str] | None = None
        self._location_resolved = False
        self._location_checked_at: float | None = None
        self._cached_context: str | None = None
        self._cached_at = 0.0

    def current_context(self) -> str:
        if not self._city:
            return self._format_time(datetime.now())

        now = time.monotonic()
        if self._cached_context is not None and now - self._cached_at < CACHE_SECONDS:
            return self._cached_context

        weather = self._fetch_weather()
        if weather is None:
            return self._format_time(datetime.now())

        observed_at, summary = weather
        context = f"{self._format_time(observed_at)}\n{summary}"
        self._cached_context = context
        self._cached_at = now
        return context

    def _resolve_location(self) -> tuple[float, float, str] | None:
        if self._location_resolved:
            return self._location

        now = time.monotonic()
        if (
            self._location_checked_at is not None
            and now - self._location_checked_at < CACHE_SECONDS
        ):
            return None

        self._location_checked_at = now
        self._location = geocode_city(self._client, self._city)
        if self._location is not None:
            self._location_resolved = True

        return self._location

    def _fetch_weather(self) -> tuple[datetime, str] | None:
        location = self._resolve_location()
        if location is None:
            return None

        latitude, longitude, name = location
        payload = fetch_forecast(self._client, latitude, longitude)
        if payload is None:
            return None

        try:
            current = payload["current"]
            observed_at = datetime.fromisoformat(current["time"])
            summary = current_line(current, name)
            summary += self._format_forecast(payload.get("daily"), payload.get("hourly"), name)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Open-Meteo returned invalid weather data: %s", exc)
            return None

        return observed_at, summary

    @staticmethod
    def _format_forecast(
        daily: dict | None,
        hourly: dict | None,
        name: str,
    ) -> str:
        if not daily:
            return ""

        slots = slots_by_date(hourly)
        lines = []
        for index in range(len(daily.get("time") or [])):
            entry = day_line(daily, index)
            if entry is None:
                continue

            date_key, line = entry
            lines.append(f"- {line}.")

            day_slots = slots.get(date_key, {})
            for slot_name in SLOT_ORDER:
                slot_line = format_slot_line(slot_name, day_slots.get(slot_name))
                if slot_line:
                    lines.append(slot_line)

        if not lines:
            return ""

        return f"\nPrevisioni a {name}:\n" + "\n".join(lines)

    @staticmethod
    def _format_time(moment: datetime) -> str:
        weekday = WEEKDAYS[moment.weekday()]
        return f"Ora e data locali: {weekday} {moment.strftime('%d/%m/%Y %H:%M')}."
