
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class AnswerStatus(str, Enum):
    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    BOTH = "both"
    INSUFFICIENT = "insufficient"


class ModelAnswer(BaseModel):
    question_id: str
    status: AnswerStatus
    answer: str | None
    evidence_ids: list[str]
    counterevidence_ids: list[str]
    confidence: float = Field(ge=0, le=1)
    explanation: str

    @model_validator(mode="after")
    def validate_answer(self):
        if self.status == AnswerStatus.BOTH:
            if not self.evidence_ids or not self.counterevidence_ids:
                raise ValueError(
                    "Для both нужны доказательства и контрдоказательства"
                )

        if self.status == AnswerStatus.INSUFFICIENT:
            if not self.explanation.strip():
                raise ValueError(
                    "Необходимо объяснить, каких данных не хватает"
                )

        return self
