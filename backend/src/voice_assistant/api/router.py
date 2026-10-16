from fastapi import APIRouter

from voice_assistant.api.routes import conversation, health, speech, transcription

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(transcription.router, tags=["transcription"])
api_router.include_router(speech.router, tags=["speech"])
api_router.include_router(conversation.router, tags=["conversation"])
