import os
from typing import TypedDict, List, BaseModel, Field

from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import InfoSQLDatabaseTool, QuerySQLDatabaseTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain.prebuilt import ToolNode
from langgraph.graph import StateGraph
from classes import Candidate, RateCandidate, RateCandidates, TopCandidates, Candidates
from nodes import get_task, generate_accents, choose_candidates,rate_candidates, return_candidates, ask_next

class QueryVariants(BaseModel):
    accents: List[str] = Field(
        ...,
        description='Описания разных вариантов, на что можно сделать акцент в ходе поиска',
    )


class State(TypedDict, total=False):
    task: str                            # исходный запрос рекрутера
    extra_task: str                      # уточнение после первой итерации
    accents: List[str]                   # сгенерированные акценты поиска
    raw_candidates: List[TopCandidates]  # кандидаты по каждому акценту
    ranked: List[CandidateScore]         # итоговый топ

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "candidates_db")

pstrgesql = 'http://postgres/'

engine = create_engine(
    f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

db = SQLDatabase(engine=engine)
info_tool = InfoSQLDatabaseTool(db=db)
query_tool = QuerySQLDatabaseTool(db=db)

llm = ChatOpenAI(
    model=os.getenv("API_MODEL"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0,
)

def fetch_candidates_by_ids(engine, ids: List[int]) -> Dict[int, dict]:
    """
    Берём из БД реальные поля по списку айдишников.
    Возвращаем словарь {id: {...факты из базы...}}
    """
    if not ids:
        return {}

    placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
    params = {f"id_{i}": cid for i, cid in enumerate(ids)}

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

    return {row["id"]: dict(row) for row in rows}

from nodes import (
    node_get_task,
    node_generate_accents,
    node_choose_candidates,
    node_rate_candidates,
    node_return_candidates,
    node_ask_next,
)

workflow = StateGraph(State)

workflow.add_node("get_task", node_get_task)
workflow.add_node("generate_accents", node_generate_accents(llm))
workflow.add_node("choose_candidates", node_choose_candidates(llm, tools_node))
workflow.add_node("rate_candidates", node_rate_candidates(llm))
workflow.add_node("return_candidates", node_return_candidates(fetch_candidates_by_ids))
workflow.add_node("ask_next", node_ask_next)

workflow.add_edge("get_task", "generate_accents")
workflow.add_edge("generate_accents", "choose_candidates")
workflow.add_edge("choose_candidates", "rate_candidates")
workflow.add_edge("rate_candidates", "return_candidates")
workflow.add_edge("return_candidates", "ask_next")
workflow.add_edge("ask_next", "generate_accents")

app = workflow.compile()