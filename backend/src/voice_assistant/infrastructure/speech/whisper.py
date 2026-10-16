from faster_whisper import WhisperModel

from voice_assistant.core.config import Settings


class WhisperRecognizer:
    def __init__(self, settings: Settings) -> None:
        self._beam_size = settings.whisper_beam_size
        self._model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            cpu_threads=settings.whisper_cpu_threads,
        )

    def transcribe(self, audio_path: str) -> dict:
        segments, info = self._model.transcribe(
            audio_path,
            beam_size=self._beam_size,
            vad_filter=True,
        )

        # Consume the lazy generator before returning the result.
        segments = list(segments)

        return {
            "text": "".join(segment.text for segment in segments).strip(),
            "language": info.language,
            "duration": info.duration,
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                for segment in segments
            ],
        }