from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Candidate(BaseModel):
    id: str = Field(..., description="ID кандидата в таблице candidates")
    desired_position: Optional[str] = Field(
        None,
        description="Желаемая должность / позиция кандидата",
    )
    city: Optional[str] = Field(
        None,
        description="Город кандидата",
    )
    work_experience: Optional[str] = Field(
        None,
        description="Краткое описание релевантного опыта, как видит его модель",
    )
    expected_salary_rub: Optional[int] = Field(
        None,
        description="Желаемая зарплата кандидата в рублях (если известна)",
    )


class CandidateScore(BaseModel):
    candidate: Candidate = Field(
        ..., description="Кандидат (как его описала модель)"
    )
    score: float = Field(
        ...,
        description="Насколько кандидат подходит запросу (0..10)",
    )
    reason: str = Field(
        ...,
        description="Почему модель считает, что кандидат подходит",
    )


class TopCandidates(BaseModel):
    accent: str = Field(..., description="Акцент (вариант формулировки запроса)")
    candidates: List[CandidateScore] = Field(
        ...,
        description="Список кандидатов с оценками по данному акценту",
    )