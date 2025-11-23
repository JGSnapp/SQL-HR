import os
from typing import TypedDict, List, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

import logging
logging.basicConfig(level=logging.INFO)

# Наши схемы и узлы
from candidates import CandidateScore, TopCandidates
from nodes import (
    node_get_task,
    node_generate_accents,
    node_choose_candidates,
    node_add_candidates,
    node_rate_candidates,
    node_return_candidates,
    node_ask_next,
    node_test_db,
)

# -----------------------------
# Состояние графа (как у тебя)
# -----------------------------
class QueryVariants(BaseModel):
    accents: List[str] = Field(
        ...,
        description="Описания разных вариантов, на что можно сделать акцент в ходе поиска",
    )

class State(TypedDict, total=False):
    task: str                            # исходный запрос рекрутера
    extra_task: str                      # уточнение после первой итерации
    accents: List[str]                   # сгенерированные акценты поиска
    raw_candidates: List[TopCandidates]  # кандидаты по каждому акценту
    ranked: List[CandidateScore]         # итоговый топ

# -----------------------------
# Подключение к БД
# -----------------------------
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "candidates_db")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

# ---------------------------------
# LLM (как у тебя через переменные)
# ---------------------------------

LLM_MODEL = os.getenv("LLM_MODEL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8010/v1")

llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,
    )

# ----------------------------------------------------------
# Утилита: получить карточки по списку id (оставил как было)
# ----------------------------------------------------------
def fetch_candidates_by_ids(engine, ids: List[UUID | str]) -> Dict[str, dict]:
    """
    Берём из БД реальные поля по списку айдишников.
    Возвращаем словарь {id: {...факты из базы...}}
    """
    if not ids:
        return {}

    placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
    params = {f"id_{i}": str(cid) for i, cid in enumerate(ids)}

    query = text(f"""
        SELECT
            id,
            sex,
            expected_salary_rub,
            desired_position,
            city,
            ready_to_relocate,
            ready_for_business_trips,
            employment_type,
            work_schedule,
            work_experience,
            last_company,
            last_job_title,
            education_level_and_university,
            resume_updated_at,
            has_car
        FROM candidates
        WHERE id IN ({placeholders})
    """)

    with engine.connect() as conn:
        rows = conn.execute(query, params).mappings().all()

    return {str(row["id"]): dict(row) for row in rows}

# ----------------------------------------------------------
# Обёртки-узлы для LangGraph (передаём llm и Session корректно)
# ----------------------------------------------------------
def _node_generate_accents(state: State) -> State:
    return node_generate_accents(state, llm)

def _node_choose_candidates(state: State) -> State:
    # на каждый вызов узла открываем короткую сессию
    with Session(bind=engine, expire_on_commit=False) as s:
        return node_choose_candidates(state, llm, s)
    
def _node_add_candidates(state: State) -> State:
    with Session(bind=engine, expire_on_commit=False) as s:
        return node_add_candidates(
            state, llm, s,
            target_n=15,     # ← сколько хотим на акцент (поменяйте под себя)
            batch_limit=30,  # ← размер batch из БД за итерацию
            max_iters=3,     # ← ограничитель, чтобы не крутиться бесконечно
        )

def _node_rate_candidates(state: State) -> State:
    # ранжирование через LLM и NormIDs (топ-10)
    return node_rate_candidates(state, llm, top_n=10)

def _node_return_candidates(state: State) -> State:
    # текущая реализация узла печатает state["ranked"], fetch_* не требуется
    # если захочешь: можно обогатить карточки из БД, вызвав fetch_candidates_by_ids
    return node_return_candidates(state)


def _node_test_db(state: State) -> State:
    with Session(bind=engine, expire_on_commit=False) as s:
        return node_test_db(state, s)


# -----------------------------
# Сборка графа
# -----------------------------
workflow = StateGraph(State)

workflow.add_node("test_db", _node_test_db)
workflow.add_node("get_task", node_get_task)
workflow.add_node("generate_accents", _node_generate_accents)
workflow.add_node("choose_candidates", _node_choose_candidates)
workflow.add_node("add_candidates", _node_add_candidates)
workflow.add_node("rate_candidates", _node_rate_candidates)
workflow.add_node("return_candidates", _node_return_candidates)
workflow.add_node("ask_next", node_ask_next)

workflow.set_entry_point("test_db")

workflow.add_edge("test_db", "get_task")
workflow.add_edge("get_task", "generate_accents")
workflow.add_edge("generate_accents", "choose_candidates")
workflow.add_edge("choose_candidates", "add_candidates")
workflow.add_edge("add_candidates", "rate_candidates")
workflow.add_edge("choose_candidates", "rate_candidates")
workflow.add_edge("rate_candidates", "return_candidates")
workflow.add_edge("return_candidates", "ask_next")


app = workflow.compile()

# ----------------------------------------------------------
# (опционально) Простой запуск из CLI:
# ----------------------------------------------------------
if __name__ == "__main__":
    # Пустое состояние — граф сам запросит задачу и пойдёт по циклу
    state: State = {}
    # Один проход цикла:
    state = app.invoke(state)
    # Если хочешь множественные итерации — вызывай несколько раз или оставь как есть (узел ask_next зацикливает граф)
