from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict

# --- SQLAlchemy ORM base & types ---
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import Text, Integer, Boolean, TIMESTAMP, text as sql_text


class Base(DeclarativeBase):
    """Declarative base for ORM models.

    Если у вас уже есть свой Base в проекте, можно удалить этот класс
    и импортировать его оттуда. Здесь оставлен для самодостаточности файла.
    """


class CandidateORM(Base):
    __tablename__ = "candidates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=sql_text("gen_random_uuid()"),  # соответствует schema.sql
    )

    sex: Mapped[Optional[str]] = mapped_column(Text)
    expected_salary_rub: Mapped[Optional[int]] = mapped_column(Integer)
    desired_position: Mapped[Optional[str]] = mapped_column(Text)
    city: Mapped[Optional[str]] = mapped_column(Text)
    ready_to_relocate: Mapped[Optional[bool]] = mapped_column(Boolean)
    ready_for_business_trips: Mapped[Optional[bool]] = mapped_column(Boolean)
    employment_type: Mapped[Optional[str]] = mapped_column(Text)
    work_schedule: Mapped[Optional[str]] = mapped_column(Text)
    work_experience: Mapped[Optional[str]] = mapped_column(Text)
    last_company: Mapped[Optional[str]] = mapped_column(Text)
    last_job_title: Mapped[Optional[str]] = mapped_column(Text)
    education_level_and_university: Mapped[Optional[str]] = mapped_column(Text)
    resume_updated_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=False))
    has_car: Mapped[Optional[bool]] = mapped_column(Boolean)


# --- Pydantic schemas (JSON-friendly) ---

class CandidateOut(BaseModel):
    """Публичное представление кандидата (готово к JSON/LLM инструментам)."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    sex: Optional[str] = None
    expected_salary_rub: Optional[int] = None
    desired_position: Optional[str] = None
    city: Optional[str] = None
    ready_to_relocate: Optional[bool] = None
    ready_for_business_trips: Optional[bool] = None
    employment_type: Optional[str] = None
    work_schedule: Optional[str] = None
    work_experience: Optional[str] = None
    last_company: Optional[str] = None
    last_job_title: Optional[str] = None
    education_level_and_university: Optional[str] = None
    resume_updated_at: Optional[datetime] = None
    has_car: Optional[bool] = None


# Для совместимости с прежним импортом: Candidate == CandidateOut
Candidate = CandidateOut


class CandidateScore(BaseModel):
    """Оценённый кандидат. Здесь оставляем только approved (как в твоём наброске)."""
    candidate: CandidateOut = Field(..., description="Кандидат (как JSON-схема)")
    approved: bool = Field(..., description="Соответствует ли кандидат запросу")


class TopCandidates(BaseModel):
    accent: str = Field(..., description="Акцент (вариант формулировки запроса)")
    candidates: List[CandidateScore] = Field(
        ..., description="Список кандидатов с флагом соответствия по данному акценту"
    )


class NormIDs(BaseModel):
    """Структурированный ответ LLM с нормализованными/выбранными id кандидатов."""
    candidates: List[uuid.UUID] = Field(
        ..., description="Список id кандидатов, которые подходят под запрос"
    )


__all__ = [
    "Base",
    "CandidateORM",
    "CandidateOut",
    "Candidate",
    "CandidateScore",
    "TopCandidates",
    "NormIDs",
]
