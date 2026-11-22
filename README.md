# Voice Assistant

A small HTTP service that turns spoken audio into spoken answers: Whisper transcribes the audio, DeepSeek generates a reply, and Piper synthesizes it back to WAV.

Built with [FastAPI](https://fastapi.tiangolo.com/), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Piper](https://github.com/rhasspy/piper) and the OpenAI-compatible DeepSeek API.

## Stack

- Python >= 3.12
- FastAPI + Uvicorn
- faster-whisper (speech-to-text, CPU by default)
- piper-tts (text-to-speech)
- openai SDK (DeepSeek, OpenAI-compatible endpoint)
- pydantic-settings for configuration

## Layout

```
backend/
├── Dockerfile
├── pyproject.toml
├── tests/unit/                     # pytest unit tests (external services mocked)
└── src/voice_assistant/
    ├── main.py                     # FastAPI app and lifespan (loads all models/services)
    ├── api/routes/                 # HTTP endpoints
    ├── services/                   # orchestration, conversation state, synthesis, LLM wrapper
    ├── infrastructure/
    │   ├── speech/                 # faster-whisper and Piper adapters
    │   └── llm/                    # DeepSeek provider
    ├── schemas/                    # Pydantic request/response models
    └── core/                       # configuration and logging
```

A request flows through `api/routes` → `services` → `infrastructure` → `schemas`.

## Local setup

All commands are run from `backend/`.

```bash
cd backend

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e '.[dev]'
```

Download a Piper voice model (`.onnx` file) from the [Piper voices](https://huggingface.co/rhasspy/piper-voices) collection and point `PIPER_MODEL_PATH` at it. Whisper weights are downloaded on first startup and cached; set `HF_HOME` to keep them between runs:

```bash
export HF_HOME="$PWD/models"
```

## Configuration

Settings are read from environment variables or a `.env` file in the working directory (names are case-insensitive). The service will not start if a required setting is missing or the Piper model file does not exist.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `PIPER_MODEL_PATH` | yes | – | Path to the Piper `.onnx` voice model |
| `DEEPSEEK_API_KEY` | yes | – | DeepSeek API key |
| `DEEPSEEK_BASE_URL` | yes | – | DeepSeek (or compatible) API base URL |
| `DEEPSEEK_MODEL` | yes | – | Chat model name, e.g. `deepseek-chat` |
| `DEEPSEEK_TIMEOUT` | no | `30.0` | Request timeout in seconds |
| `DEEPSEEK_MAX_RETRIES` | no | `2` | SDK retry count |
| `WHISPER_MODEL` | no | `base` | Whisper model name or path (`tiny`, `base`, `small`, `medium`, `large-v3`, ...) |
| `WHISPER_DEVICE` | no | `cpu` | Execution device (`cpu`, `cuda`) |
| `WHISPER_COMPUTE_TYPE` | no | `int8` | Compute precision (`int8`, `float16`, `float32`, ...) |
| `WHISPER_BEAM_SIZE` | no | `1` | Decoding beam size |
| `WHISPER_CPU_THREADS` | no | `4` | CPU threads used by Whisper |
| `CONVERSATION_MAX_TURNS` | no | `5` | Exchanges kept per device |
| `CONVERSATION_TTL_SECONDS` | no | `600` | Inactivity before a device session expires |

Example `.env`:

```dotenv
PIPER_MODEL_PATH=/models/it_IT-paola-medium.onnx
DEEPSEEK_API_KEY=your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

Settings are cached: restart the service after changing the environment.

## Running

```bash
cd backend
uvicorn voice_assistant.main:app --reload
```

To accept connections from other machines on the network, e.g. an ESP32 device, bind to all interfaces:

```bash
cd backend
uvicorn voice_assistant.main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` exposes the service on every network interface, so make sure the firewall/network allow port `8000` only where intended. Add `--reload` for development if needed.

The server listens on port `8000`. Interactive docs: `http://localhost:8000/docs`. The Docker image already starts Uvicorn with `--host 0.0.0.0`.

## API

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /transcribe`

Transcribes an audio file and returns JSON. Expects a file in `multipart/form-data` under the `audio` field.

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@sample.wav"
```

```json
{
  "text": "Hello, this is a test transcription.",
  "language": "en",
  "duration": 4.2,
  "segments": [
    { "start": 0.0, "end": 4.2, "text": "Hello, this is a test transcription." }
  ]
}
```

Errors: `400` if the uploaded file is empty, `500` if transcription fails.

### `POST /synthesize`

Converts text to speech and returns a WAV file (`audio/wav`). JSON body, `text` must be 1–2000 characters.

```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Ciao, come posso aiutarti?"}' \
  --output speech.wav
```

Errors: `422` for an invalid payload, `500` if synthesis fails.

### `POST /conversation`

Full voice loop for a device: transcribes the uploaded audio, keeps per-device conversation history, asks the LLM, and returns the synthesized answer as a WAV file.

- `audio`: audio file (`multipart/form-data`)
- `esp_id`: device identifier (`multipart/form-data`), used to scope the conversation

```bash
curl -X POST http://localhost:8000/conversation \
  -F "audio=@question.wav" \
  -F "esp_id=esp32-living-room" \
  --output response.wav
```

The LLM is instructed to answer in Italian, concisely and without Markdown, since the output is spoken.

Conversation state is in memory, per process: the last `CONVERSATION_MAX_TURNS` exchanges per `esp_id`, dropped after `CONVERSATION_TTL_SECONDS` of inactivity. It is lost on restart and not shared across multiple workers.

Errors: `400` for an empty upload or blank `esp_id`, `422` when no speech is detected in the audio or the request is invalid, `500` if any pipeline step fails.

### `POST /conversation/text`

Text-only variant of `/conversation`: no transcription and no synthesis, useful for testing the LLM and conversation history without audio. It shares the same per-device history as the audio endpoint.

JSON body:

- `esp_id`: device identifier
- `message`: user text, 1–2000 characters

```bash
curl -X POST http://localhost:8000/conversation/text \
  -H "Content-Type: application/json" \
  -d '{"esp_id": "esp32-living-room", "message": "Che tempo fa?"}'
```

```json
{ "reply": "Non ho accesso al meteo in tempo reale, ma posso aiutarti con altro." }
```

Errors: `422` for a blank `esp_id`/`message` or invalid payload, `500` if the LLM call fails.

## Docker

The image does not bundle a Piper voice, so mount the `.onnx` model and pass the required settings:

```bash
cd backend
docker build -t voice-assistant .

docker run --rm -p 8000:8000 \
  -v whisper-models:/models \
  -v /path/to/it_IT-paola-medium.onnx:/models/piper.onnx:ro \
  -e PIPER_MODEL_PATH=/models/piper.onnx \
  -e DEEPSEEK_API_KEY=your-key \
  -e DEEPSEEK_BASE_URL=https://api.deepseek.com \
  -e DEEPSEEK_MODEL=deepseek-chat \
  voice-assistant
```

The container uses `HF_HOME=/models`, so the named volume caches Whisper weights across runs.

## Lint

```bash
cd backend
ruff check src
```

## Tests

Unit tests live in `backend/tests/unit` and mock Whisper, Piper, and the DeepSeek client, so they need no models, API key, network, or `.env`. The editable install is required because the package lives under `src/`:

```bash
cd backend
pip install -e '.[dev]'      # once; alternatively run with PYTHONPATH=src

pytest -v                     # whole suite
pytest tests/unit             # unit tests only
pytest tests/unit/test_llm_service.py                      # single file
pytest tests/unit/test_llm_service.py::test_generate_response_delegates_to_provider  # single test
pytest -k orchestrator        # filter by name
```
