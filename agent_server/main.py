from __future__ import annotations

import os
import uuid
import logging
import re
from typing import TypedDict, List, Dict, Optional, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from fastapi import FastAPI

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import tool

# наши модели и узлы
from candidates import CandidateScore, TopCandidates
from nodes import (
    node_generate_accents,
    node_choose_candidates,
    node_add_candidates,
    node_rate_candidates,
    node_return_candidates,
    node_ask_next,
    node_test_db,
)

# -----------------------------
# Логгер
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Тип состояния графа кандидатов
# -----------------------------
class CandidateState(TypedDict, total=False):
    task: str
    extra_task: str
    accents: List[str]
    raw_candidates: List[TopCandidates]
    ranked: List[CandidateScore]


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


# -----------------------------
# LLM
# -----------------------------
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8010/v1")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    temperature=0,
    max_tokens=LLM_MAX_TOKENS,
)


# ----------------------------------------------------------
# Утилита: получить карточки по списку id из БД
# ----------------------------------------------------------
def fetch_candidates_by_ids(engine, ids: List[UUID | str]) -> Dict[str, Dict[str, Any]]:
    """
    Берём из БД реальные поля по списку айдишников.
    Возвращаем словарь {id: {...факты из базы...}}
    """
    if not ids:
        return {}

    placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
    params = {f"id_{i}": str(cid) for i, cid in enumerate(ids)}

    query = text(
        f"""
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
    """
    )

    with engine.connect() as conn:
        rows = conn.execute(query, params).mappings().all()

    return {str(row["id"]): dict(row) for row in rows}


# ----------------------------------------------------------
# Обёртки-узлы для LangGraph (передаём llm и Session)
# ----------------------------------------------------------
def _node_generate_accents(state: CandidateState) -> CandidateState:
    return node_generate_accents(state, llm)


def _node_choose_candidates(state: CandidateState) -> CandidateState:
    with Session(bind=engine, expire_on_commit=False) as s:
        return node_choose_candidates(state, llm, s)


def _node_add_candidates(state: CandidateState) -> CandidateState:
    with Session(bind=engine, expire_on_commit=False) as s:
        return node_add_candidates(
            state,
            llm,
            s,
            target_n=15,  # сколько хотим на акцент
            batch_limit=30,  # размер batch из БД за итерацию
            max_iters=3,  # ограничитель итераций
        )


def _node_rate_candidates(state: CandidateState) -> CandidateState:
    return node_rate_candidates(state, llm, top_n=10)


def _node_return_candidates(state: CandidateState) -> CandidateState:
    # node_return_candidates сам пишет result.txt, но нам важен state["ranked"]
    return node_return_candidates(state)


def _node_test_db(state: CandidateState) -> CandidateState:
    with Session(bind=engine, expire_on_commit=False) as s:
        return node_test_db(state, s)


# -----------------------------
# Граф подбора кандидатов
# -----------------------------
candidate_workflow = StateGraph(CandidateState)

candidate_workflow.add_node("test_db", _node_test_db)
candidate_workflow.add_node("generate_accents", _node_generate_accents)
candidate_workflow.add_node("choose_candidates", _node_choose_candidates)
candidate_workflow.add_node("add_candidates", _node_add_candidates)
candidate_workflow.add_node("rate_candidates", _node_rate_candidates)
candidate_workflow.add_node("return_candidates", _node_return_candidates)
candidate_workflow.add_node("ask_next", node_ask_next)

candidate_workflow.set_entry_point("test_db")

candidate_workflow.add_edge("test_db", "generate_accents")
candidate_workflow.add_edge("generate_accents", "choose_candidates")
candidate_workflow.add_edge("choose_candidates", "add_candidates")
candidate_workflow.add_edge("add_candidates", "rate_candidates")
candidate_workflow.add_edge("choose_candidates", "rate_candidates")
candidate_workflow.add_edge("rate_candidates", "return_candidates")
candidate_workflow.add_edge("return_candidates", "ask_next")

candidate_graph = candidate_workflow.compile()


# -----------------------------
# Тул для LLM: choose_candidates
# -----------------------------
@tool
def choose_candidates(query: str) -> List[str]:
    """
    Инструмент для LLM: запускает граф подбора кандидатов и
    возвращает список id найденных кандидатов (строками).

    LLM видит только список id, а полные данные мы достаём на бэке.
    """
    state: CandidateState = candidate_graph.invoke({"task": query})
    ranked: List[CandidateScore] = state.get("ranked", [])  # может отсутствовать
    ids = [str(item.candidate.id) for item in ranked]
    logger.info("choose_candidates tool: picked %d candidates", len(ids))
    return ids


# -----------------------------
# Граф-агент с tool calling
# -----------------------------
def add_messages(left: List[AnyMessage], right: List[AnyMessage]) -> List[AnyMessage]:
    return left + right


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


tools = [choose_candidates]
tool_node = ToolNode(tools)


def agent_node(state: AgentState) -> AgentState:
    """LLM-агент, умеющий вызывать инструменты."""
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def router(state: AgentState) -> Literal["tools", "end"]:
    """Маршрутизатор: решает, нужно ли идти в ToolNode."""
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return "end"


main_workflow = StateGraph(AgentState)
main_workflow.add_node("agent", agent_node)
main_workflow.add_node("tools", tool_node)

main_workflow.set_entry_point("agent")
main_workflow.add_conditional_edges(
    "agent",
    router,
    {
        "tools": "tools",
        "end": END,
    },
)
# после tools всегда возвращаемся к agent за финальным ответом
main_workflow.add_edge("tools", "agent")

graph_app = main_workflow.compile()


# -----------------------------
# FastAPI-приложение
# -----------------------------
app = FastAPI(title="SQL-HR Chat API")

# Память диалогов: session_id -> история сообщений LangChain
SESSIONS: Dict[str, List[AnyMessage]] = {}

CHAT_SYSTEM_PROMPT = (
    "Ты - HR-помощник. Твоя задача — помочь пользователю максимально чётко "
    "сформулировать запрос к кандидату. "
    "Когда человек просит подобрать кандидатов (или информации достаточно), "
    "вызови инструмент choose_candidates. "
    "Говори по-русски, дружелюбно, без лишней воды."
)

UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


class ChatRequest(BaseModel):
    # Клиент МОЖЕТ не прислать session_id (первый запрос) —
    # тогда мы сгенерируем его сами.
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    # Всегда возвращаем session_id, чтобы клиент мог его сохранить
    session_id: str
    answer: str
    # id -> данные кандидата из БД
    candidates: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


def _extract_candidate_ids(messages: List[AnyMessage]) -> List[str]:
    """Достаём id из свежих ToolMessage choose_candidates в этом прогоне."""
    ids: List[str] = []
    seen: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "choose_candidates":
            content = msg.content
            if isinstance(content, list):
                for x in content:
                    sx = str(x)
                    if sx not in seen:
                        ids.append(sx)
                        seen.add(sx)
    return ids


def _extract_tool_messages(messages: List[AnyMessage]) -> List[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


@app.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid.uuid4())
    logger.info("chat: session=%s incoming=%s", session_id, req.message)
    if session_id not in SESSIONS:
        SESSIONS[session_id] = [SystemMessage(content=CHAT_SYSTEM_PROMPT)]

    history = SESSIONS[session_id]

    history.append(HumanMessage(content=req.message))

    state: AgentState = graph_app.invoke({"messages": history})
    all_messages = state["messages"]

    ai_msgs = [m for m in all_messages if isinstance(m, AIMessage)]
    answer = ai_msgs[-1].content if ai_msgs else ""

    # сохраняем tool_messages и ai_messages из текущего прогона в историю для следующего шага
    tool_msgs = _extract_tool_messages(all_messages)
    history.extend(tool_msgs)
    if ai_msgs:
        history.append(ai_msgs[-1])

    # --- 6. Если агент вызывал choose_candidates — достаём id ---
    candidate_ids = _extract_candidate_ids(all_messages)
    candidate_profiles = fetch_candidates_by_ids(engine, candidate_ids)
    logger.info(
        "chat: session=%s extracted_ids=%s profiles=%d",
        session_id,
        candidate_ids,
        len(candidate_profiles),
    )

    # --- 7. Возвращаем и answer, и session_id, и кандидатов ---
    return ChatResponse(
        session_id=session_id,
        answer=answer,
        candidates=candidate_profiles,
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    """Простая проверка готовности сервиса."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
