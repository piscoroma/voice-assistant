from contextlib import asynccontextmanager

from fastapi import FastAPI

from voice_assistant.api.router import api_router
from voice_assistant.core.config import get_settings
from voice_assistant.core.logging import setup_logging
from voice_assistant.infrastructure.llm.deepseek import DeepSeekProvider
from voice_assistant.infrastructure.speech.piper import PiperSynthesizer
from voice_assistant.infrastructure.speech.whisper import WhisperRecognizer
from voice_assistant.infrastructure.weather.open_meteo import OpenMeteoContext
from voice_assistant.services.conversation import ConversationService
from voice_assistant.services.conversation_orchestrator import ConversationOrchestrator
from voice_assistant.services.llm import LLMService
from voice_assistant.services.speech_synthesis import SpeechSynthesisService
from voice_assistant.services.transcription import TranscriptionService
from voice_assistant.tools.base import Tool
from voice_assistant.tools.weather import build_weather_tool


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    recognizer = WhisperRecognizer(settings)
    synthesizer = PiperSynthesizer(settings)

    tools: list[Tool] = [build_weather_tool(settings.assistant_city)]

    deepseek = DeepSeekProvider(
        settings,
        context_provider=OpenMeteoContext(settings.assistant_city).current_context,
        tools=tools,
    )
    app.state.transcription_service = TranscriptionService(recognizer)
    app.state.speech_synthesis_service = SpeechSynthesisService(synthesizer)
    app.state.llm_service = LLMService(deepseek)
    app.state.conversation_service = ConversationService(
        max_turns=settings.conversation_max_turns, ttl_seconds=settings.conversation_ttl_seconds
    )
    app.state.conversation_orchestrator = ConversationOrchestrator(
        transcription_service=app.state.transcription_service,
        conversation_service=app.state.conversation_service,
        speech_synthesis_service=app.state.speech_synthesis_service,
        llm_service=app.state.llm_service,
    )

    yield

    # clean resources


app = FastAPI(
    title="Voice Assistant API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api_router)
