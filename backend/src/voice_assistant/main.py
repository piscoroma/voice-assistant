from contextlib import asynccontextmanager

from fastapi import FastAPI

from voice_assistant.api.router import api_router
from voice_assistant.core.config import get_settings
from voice_assistant.core.logging import setup_logging
from voice_assistant.infrastructure.speech.whisper import WhisperRecognizer
from voice_assistant.services.transcription import TranscriptionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    recognizer = WhisperRecognizer(settings)
    app.state.transcription_service = TranscriptionService(recognizer)

    yield

    # clean resources


app = FastAPI(
    title="Voice Assistant API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api_router)