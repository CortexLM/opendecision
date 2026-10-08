"""Wire format. Laya compatible: {state, questions:{id:{type,instructions,criteria}}}.
v1 is TEXT ONLY. Media (image/audio/video) comes in a later version; unknown fields are rejected."""
from __future__ import annotations
from typing import Literal, Union
from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

QType = Literal["choice", "score", "noul"]


class Question(BaseModel):
    type: QType
    instructions: StrictStr
    # choice: {option: description}; score: ordered level descriptions; noul: unused
    criteria: Union[dict[str, str], list[str], None] = None

    @model_validator(mode="after")
    def _check(self):
        if self.type == "choice" and not (isinstance(self.criteria, dict) and len(self.criteria) >= 2):
            raise ValueError("choice needs criteria dict with >=2 options")
        if self.type == "score" and not (isinstance(self.criteria, list) and len(self.criteria) >= 2
                                         and all(isinstance(c, str) and c for c in self.criteria)):
            raise ValueError("score needs >=2 non-empty level descriptions")
        return self

    def options(self) -> list[str]:
        if self.type == "choice":
            return list(self.criteria)
        if self.type == "score":
            return list(self.criteria)
        return ["no", "yes"]



class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Union[StrictStr, dict]
    questions: dict[str, Question] = Field(min_length=0)
    model: str | None = None
