# AGENTS.md

Python/FastAPI voice pipeline (Whisper STT → DeepSeek LLM → Piper TTS). The only code is in `backend/`; there is no CI.

## Commands (run from `backend/`)

```bash
pip install -e '.[dev]'                # Python >= 3.12; editable install makes `voice_assistant` importable
uvicorn voice_assistant.main:app --reload
ruff check src                        # line-length is 100 (pyproject)
pytest                                 # unit tests in tests/unit
pytest tests/unit/test_llm_service.py::test_generate_response_delegates_to_provider
docker build -t voice-assistant .
```

Tests are plain pytest unit tests with mocked Whisper/Piper/OpenAI: no models, API key, network, or `.env` needed. They import `voice_assistant`, so the editable install (or `PYTHONPATH=src`) is required; without it collection fails with `ModuleNotFoundError`.

Running `pytest`/`ruff`/`uvicorn` from the repo root will fail: there is no root manifest and the package lives under `backend/src/voice_assistant`. `rsync.sh` deploys the tree to a homelab host; it is not part of the app.

## Architecture

Request flow: `api/routes/*` (thin HTTP handlers) → `services/` → `infrastructure/` (Whisper, Piper, DeepSeek adapters) → `schemas/` (Pydantic models). Settings and logging live in `core/`.

- All model/service instances are built once in the FastAPI lifespan in `main.py` and hung on `app.state` (`transcription_service`, `speech_synthesis_service`, `llm_service`, `conversation_service`, `conversation_orchestrator`). Routes read them from there; don't construct models per request.
- `ConversationOrchestrator` chains transcribe → history lookup → LLM → synthesis and returns a WAV `Path`. `process_text()` is the text-only path (`/conversation/text`) with the same history/LLM logic but no STT/TTS.
- `ConversationService` keeps history in memory keyed by `esp_id` (bounded deque plus TTL). State is per process, lost on restart, and not shared across workers.
- `OpenMeteoContext` is passed to `DeepSeekProvider` as `context_provider`; it resolves `ASSISTANT_CITY` via Open-Meteo geocoding and returns local time, current weather and a 3-day forecast with night/morning/afternoon/evening slots, cached for 10 minutes. Failures degrade to time-only text (no exception).
- `tools/base.py` defines `Tool` (name, description, JSON schema, handler). `DeepSeekProvider` runs an OpenAI-style tool loop (max 3 rounds) and executes tools in-process; `tools/web_search.py` uses `infrastructure/web/deepseek_search.py`, and `tools/weather.py` reuses the Open-Meteo helpers from `infrastructure/weather/open_meteo.py`. Add new tools there and register them in `main.py`. Tool calls are ephemeral and are not saved in conversation history.
- Weather is hybrid: `OpenMeteoContext` injects a compact summary for `ASSISTANT_CITY`, while the `get_weather` tool covers other cities and missing detail. Both go through the same module-level geocoding/forecast/formatting helpers in `open_meteo.py`; keep them in sync instead of duplicating logic.
- `/transcribe` returns JSON; `/synthesize` and `/conversation` return WAV via `FileResponse` with a `BackgroundTask` that deletes the temp file. `/conversation/text` returns JSON and shares history with `/conversation`.

## Configuration and startup requirements

pydantic-settings reads `.env` or case-insensitive env vars. Required with no defaults: `PIPER_MODEL_PATH` (must be an existing `.onnx`), `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`. Others: `WHISPER_MODEL` (`base`), `WHISPER_DEVICE` (`cpu`), `WHISPER_COMPUTE_TYPE` (`int8`), `WHISPER_BEAM_SIZE` (`1`), `WHISPER_CPU_THREADS` (`4`), `DEEPSEEK_TIMEOUT` (`30.0`), `DEEPSEEK_MAX_RETRIES` (`2`), `CONVERSATION_MAX_TURNS` (`5`), `CONVERSATION_TTL_SECONDS` (`600`), `ASSISTANT_CITY` (empty; geocoded for time/weather context), `WEB_SEARCH_ENABLED` (`true`). `get_settings()` is `lru_cache`d, so env changes need a restart.

Startup downloads Whisper weights (slow, needs network) and loads the Piper model eagerly; missing settings abort startup. Set `HF_HOME` to a persistent path to cache weights; `backend/models/` is gitignored and `.dockerignore`d, so Docker needs a volume.

## Gotchas

- The whole pipeline is CPU-bound and blocking. Handlers are plain `def` so FastAPI runs them in a threadpool; keep it that way rather than converting to `async def`.
- faster-whisper's `transcribe()` returns a lazy generator; it must be fully consumed (`list(segments)`) before returning, or transcription never runs.
- Uploaded audio is copied to a temp file and removed in `finally`; synthesized WAVs are removed by the response background task. Preserve these cleanups.
- The Dockerfile copies only `pyproject.toml` and `src/`, and bundles no Piper voice: mount the `.onnx` and set `PIPER_MODEL_PATH`.
- `av<19` is an intentional upper pin; don't loosen it without verifying faster-whisper decoding.
- Weather/time lookups are synchronous HTTP calls made inside the sync request handler; keep tests offline with `httpx.MockTransport` (see `tests/unit/test_open_meteo_context.py`).
- The tool loop caps at `MAX_TOOL_ROUNDS` (3) and raises if the model keeps calling tools; tool exceptions are returned to the model as text instead of failing the turn. Successful executions log tool name, arguments (truncated) and duration at INFO. DeepSeek Search API calls are billed per successful request (~$0.00075 each), so keep tests mocked.
- Don't mix DeepSeek's server-side `deepseek_web_search` tool with custom function tools: the API rejects it. This project uses its own `web_search` function tool instead.
- The DeepSeek system prompt forces Italian, concise, Markdown-free replies because the answer is spoken aloud.
