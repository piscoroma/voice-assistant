import logging
import wave
from pathlib import Path

from piper import PiperVoice

from voice_assistant.core.config import Settings

logger = logging.getLogger(__name__)


class PiperSynthesizer:
    def __init__(self, settings: Settings) -> None:
        model_path = settings.piper_model_path

        if not model_path.is_file():
            raise FileNotFoundError(f"Piper model not found: {model_path}")

        logger.info("Loading Piper model: %s", model_path)
        self._voice = PiperVoice.load(str(model_path))
        logger.info("Piper model loaded")

    def synthesize(self, text: str, output_path: Path) -> None:
        logger.info("Starting speech synthesis")

        with wave.open(str(output_path), "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)

        logger.info(
            "Speech synthesis completed: output=%s",
            output_path,
        )
