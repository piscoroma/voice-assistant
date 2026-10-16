from fastapi import APIRouter

from voice_assistant.api.routes import health, transcription

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(transcription.router, tags=["transcription"])