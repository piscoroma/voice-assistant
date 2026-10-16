import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from voice_assistant.schemas.transcription import TranscriptionResponse
from voice_assistant.services.transcription import TranscriptionService

router = APIRouter()


@router.post("/transcribe", response_model=TranscriptionResponse)
def transcribe(
    request: Request,
    audio: UploadFile = File(...),
) -> TranscriptionResponse:
    service: TranscriptionService = request.app.state.transcription_service

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp_path = temp.name
            shutil.copyfileobj(audio.file, temp)

        if Path(temp_path).stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio is empty")

        return service.transcribe(temp_path)

    except HTTPException:
        raise
    except Exception as exc:
        # Keep internal exception details out of the HTTP response.
        raise HTTPException(
            status_code=500,
            detail="Transcription failed",
        ) from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
        audio.file.close()