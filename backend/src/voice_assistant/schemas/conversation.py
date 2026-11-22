from pydantic import BaseModel, Field, field_validator


class ConversationTextRequest(BaseModel):
    esp_id: str
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("esp_id", "message")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ConversationTextResponse(BaseModel):
    reply: str
