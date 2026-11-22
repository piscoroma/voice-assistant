import pytest
from pydantic import ValidationError

from voice_assistant.schemas.conversation import ConversationTextRequest


def test_request_strips_whitespace():
    request = ConversationTextRequest(esp_id="  esp32-salotto  ", message="  Ciao  ")

    assert request.esp_id == "esp32-salotto"
    assert request.message == "Ciao"


@pytest.mark.parametrize(
    ("esp_id", "message"),
    [
        ("   ", "Ciao"),
        ("esp32-salotto", "   "),
    ],
)
def test_request_rejects_blank_fields(esp_id, message):
    with pytest.raises(ValidationError):
        ConversationTextRequest(esp_id=esp_id, message=message)
