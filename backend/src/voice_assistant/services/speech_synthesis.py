import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

from voice_assistant.infrastructure.speech.piper import PiperSynthesizer

logger = logging.getLogger(__name__)


class SpeechSynthesisService:
    def __init__(self, synthesizer: PiperSynthesizer) -> None:
        self._synthesizer = synthesizer

    def synthesize(self, text: str) -> Path:
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp:
            output_path = Path(temp.name)

        try:
            self._synthesizer.synthesize(text, output_path)
            return output_path
        except Exception:
            output_path.unlink(missing_ok=True)
            logger.exception("Speech synthesis failed")
            raise
