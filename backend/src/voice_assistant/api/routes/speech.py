import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from voice_assistant.schemas.speech import SpeechRequest
from voice_assistant.services.speech_synthesis import SpeechSynthesisService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/synthesize")
def synthesize(
    request: Request,
    payload: SpeechRequest,
) -> FileResponse:
    service: SpeechSynthesisService = request.app.state.speech_synthesis_service

    try:
        output_path = service.synthesize(payload.text)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Speech synthesis failed",
        ) from exc

    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename="speech.wav",
        background=BackgroundTask(
            output_path.unlink,
            missing_ok=True,
        ),
    )
