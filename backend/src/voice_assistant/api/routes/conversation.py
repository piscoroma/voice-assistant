import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from voice_assistant.schemas.conversation import (
    ConversationTextRequest,
    ConversationTextResponse,
)
from voice_assistant.services.conversation_orchestrator import ConversationOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/conversation")
def conversation(
    request: Request,
    audio: UploadFile = File(...),
    esp_id: str = Form(...),
) -> FileResponse:

    if not esp_id.strip():
        raise HTTPException(
            status_code=400,
            detail="esp_id cannot be empty",
        )

    temp_path = None

    try:
        # Save the uploaded audio to a temporary file.
        suffix = Path(audio.filename or "audio.wav").suffix or ".wav"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp_path = temp.name
            shutil.copyfileobj(audio.file, temp)

        if Path(temp_path).stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio is empty")

        # Retrieve the orchestrator initialized at application startup.
        orchestrator: ConversationOrchestrator = request.app.state.conversation_orchestrator

        # Process the conversation and obtain the generated WAV path.
        audio_path = orchestrator.process(
            audio_path=str(temp_path),
            esp_id=esp_id,
        )

        # Return the WAV file and remove it after transmission.
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename="response.wav",
            background=BackgroundTask(
                audio_path.unlink,
                missing_ok=True,
            ),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Conversation processing failed")
        raise HTTPException(
            status_code=500,
            detail="Conversation processing failed",
        ) from exc

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

        audio.file.close()


@router.post("/conversation/text", response_model=ConversationTextResponse)
def conversation_text(
    request: Request,
    payload: ConversationTextRequest,
) -> ConversationTextResponse:
    orchestrator: ConversationOrchestrator = request.app.state.conversation_orchestrator

    try:
        reply = orchestrator.process_text(
            esp_id=payload.esp_id,
            user_message=payload.message,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Text conversation processing failed")
        raise HTTPException(
            status_code=500,
            detail="Conversation processing failed",
        ) from exc

    return ConversationTextResponse(reply=reply)
