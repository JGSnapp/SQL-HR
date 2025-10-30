from __future__ import annotations

import uuid
import json
from typing import Any, Dict, List, Optional, TypedDict, Literal

from pydantic import BaseModel, Field
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import Session
from langchain_core.messages import SystemMessage, HumanMessage

# Локальный импорт ORM/схем (исправлено: без app.domain.*)
from candidates import (
    CandidateORM as C,
    CandidateOut,
    CandidateScore,
    TopCandidates,
    NormIDs,
)
RELAX_QUERY_SYSTEM_PROMPT = (
    "Тебе дан акцент (формулировка поиска) и текущая выборка кандидатов. "
    "Нужно ДОБРАТЬ ещё кандидатов: сгенерируй новую спецификацию QuerySpec, которая расширит охват, "
    "но останется релевантной: допускается смягчить salary, добавить синонимы в keywords_any, "
    "снять часть keywords_all, расширить город (агломерации/удалёнка). "
    "Всегда указывай limit (batch_limit). Верни строго JSON модели QuerySpec."
)

# --- Простые текстовые промпты по умолчанию (можно заменить своими из prompts.py) ---
GET_TASK_PROMPT = "Опиши вакансию или нужный профиль кандидата:"
GENERATE_ACCENTS_SYSTEM_PROMPT = (
    "Ты генерируешь 2–5 разных формулировок запроса ('акцентов') для поиска кандидатов. "
    "Коротко, по сути. Верни JSON со списком 'accents'."
)
CHOOSE_CANDIDATES_SYSTEM_PROMPT = (
    "Ты выделяешь структуру фильтров QuerySpec (город, зарплата, ключевые слова) из данного акцента."
)
CHOOSE_CANDIDATES_HUMAN_PROMPT_TEMPLATE = "Акцент: {accent}"
PREVIOUS_RESULTS_SYSTEM_PROMPT = "Это предыдущие найденные кандидаты."
RATE_CANDIDATES_SYSTEM_PROMPT = (
    "Выбери релевантных кандидатов и верни их id в JSON-модели NormIDs."
)
ASK_NEXT_PROMPT = "Есть ли уточнения к поиску? (Enter, чтобы пропустить): "


# --- Тип состояния графа ---
class State(TypedDict, total=False):
    task: str
    extra_task: str
    accents: List[str]
    raw_candidates: List[TopCandidates]
    ranked: List[CandidateScore]


# --- Структуры, которые LLM заполняет в узлах ---
class QuerySpec(BaseModel):
    city: Optional[str] = None
    min_salary_rub: Optional[int] = None
    max_salary_rub: Optional[int] = None
    ready_to_relocate: Optional[bool] = None
    keywords_any: List[str] = Field(default_factory=list)
    keywords_all: List[str] = Field(default_factory=list)
    keywords_not: List[str] = Field(default_factory=list)
    seniority: Optional[Literal["junior", "middle", "senior", "lead"]] = None
    limit: int = Field(30, ge=5, le=100)


class QueryVariants(BaseModel):
    accents: List[str] = Field(..., min_items=1, max_items=8)


# --- Поиск через ORM по собранному QuerySpec ---
def get_from_query(spec: QuerySpec, session: Session, top_n: int = 20) -> List[CandidateOut]:
    clauses = []
    if spec.city:
        clauses.append(C.city.ilike(f"%{spec.city}%"))
    if spec.min_salary_rub is not None:
        clauses.append(C.expected_salary_rub >= spec.min_salary_rub)
    if spec.max_salary_rub is not None:
        clauses.append(C.expected_salary_rub <= spec.max_salary_rub)
    if spec.ready_to_relocate is not None:
        clauses.append(C.ready_to_relocate == spec.ready_to_relocate)

    for kw in (spec.keywords_all or []):
        clauses.append(C.work_experience.ilike(f"%{kw}%"))

    any_clauses = [C.work_experience.ilike(f"%{kw}%") for kw in (spec.keywords_any or [])]
    if any_clauses:
        clauses.append(or_(*any_clauses))

    not_clauses = [~C.work_experience.ilike(f"%{kw}%") for kw in (spec.keywords_not or [])]
    if not_clauses:
        clauses.append(and_(*not_clauses))

    stmt = select(C)
    if clauses:
        stmt = stmt.where(and_(*clauses))
    stmt = stmt.limit(top_n)

    rows = session.scalars(stmt).all()  # list[CandidateORM]
    return [CandidateOut.model_validate(r) for r in rows]


# --- Узлы графа ---

def node_get_task(state: State) -> State:
    """Для CLI: спросить задачу у пользователя, если её ещё нет в state."""
    if not state.get("task"):
        try:
            task = input(GET_TASK_PROMPT).strip()
        except EOFError:
            task = ""
        if task:
            state["task"] = task
    return state


def node_generate_accents(state: State, llm) -> State:
    msgs = [
        SystemMessage(content=GENERATE_ACCENTS_SYSTEM_PROMPT),
        HumanMessage(content=state.get("task", "")),
    ]
    structured = llm.with_structured_output(QueryVariants)
    accents_resp = structured.invoke(msgs)
    state["accents"] = accents_resp.accents
    return state


def node_choose_candidates(state: State, llm, session: Session) -> State:
    state.setdefault("raw_candidates", [])

    for accent in state.get("accents", []):
        # 1) LLM превращает акцент в QuerySpec
        qspec_llm = llm.with_structured_output(QuerySpec)
        llm_query = qspec_llm.invoke([
            SystemMessage(content=CHOOSE_CANDIDATES_SYSTEM_PROMPT),
            HumanMessage(content=CHOOSE_CANDIDATES_HUMAN_PROMPT_TEMPLATE.format(accent=accent)),
        ])

        # 2) ORM-поиск по спецификации
        raw_candidates = get_from_query(llm_query, session=session, top_n=llm_query.limit or 20)

        # 3) Нормализация/отбор id через LLM (по желанию — можно пропустить и считать approved=True)
        norm_llm = llm.with_structured_output(NormIDs)
        msgs = [
            SystemMessage(content=RATE_CANDIDATES_SYSTEM_PROMPT),
            HumanMessage(content=f"Запрос: {state.get('task','')}"),
            HumanMessage(content="\n".join(c.model_dump_json() for c in raw_candidates)),
        ]
        try:
            norm_ids = norm_llm.invoke(msgs).candidates
            ids_set = set(norm_ids)
            ready = [CandidateScore(candidate=c, approved=(c.id in ids_set)) for c in raw_candidates]
        except Exception:
            # если LLM не справилась со структурой — отметим всех как approved
            ready = [CandidateScore(candidate=c, approved=True) for c in raw_candidates]

        state["raw_candidates"].append(TopCandidates(accent=accent, candidates=ready))

    return state

# --- Подсказка для LLM: как ослаблять запрос ---
RELAX_QUERY_SYSTEM_PROMPT = (
    "Тебе дан акцент (формулировка поиска) и текущая выборка кандидатов. "
    "Нужно ДОБРАТЬ ещё кандидатов: сгенерируй новую спецификацию QuerySpec, которая расширит охват, "
    "но останется релевантной: допускается смягчить salary, добавить синонимы в keywords_any, "
    "снять часть keywords_all, расширить город (агломерации/удалёнка). "
    "Всегда указывай limit (batch_limit). Верни строго JSON модели QuerySpec."
)

# --- Итерационное добирание кандидатов до целевого N на каждом акценте ---
def node_add_candidates(
    state: State,
    llm,
    session: Session,
    target_n: int = 15,      # сколько в итоге хотим на 1 акцент
    batch_limit: int = 30,   # сколько тянуть за итерацию из БД
    max_iters: int = 3,      # максимум итераций расширения на акцент
) -> State:
    """
    Для каждого акцента, уже имеющегося в state['raw_candidates'],
    добирает недостающих кандидатов: на каждой итерации LLM ослабляет QuerySpec,
    мы тянем batch из БД и просим LLM выбрать до недостающего количества id.
    """
    if not state.get("raw_candidates"):
        return state

    # helper: аккуратно урезать текст, чтобы не раздувать контекст LLM
    def _short(txt: Optional[str], limit: int = 600) -> Optional[str]:
        if not txt:
            return txt
        return txt if len(txt) <= limit else (txt[:limit] + "…")

    for group in state["raw_candidates"]:
        accent = group.accent

        # текущее состояние по акценту
        approved_ids: set[uuid.UUID] = {
            cs.candidate.id for cs in group.candidates if cs.approved
        }
        seen_ids: set[uuid.UUID] = {
            cs.candidate.id for cs in group.candidates
        }

        # если уже хватает — пропускаем
        if len(approved_ids) >= target_n:
            continue

        # Базовая спецификация на первую итерацию (генерим из акцента)
        qspec = llm.with_structured_output(QuerySpec).invoke([
            SystemMessage(content=CHOOSE_CANDIDATES_SYSTEM_PROMPT),
            HumanMessage(content=CHOOSE_CANDIDATES_HUMAN_PROMPT_TEMPLATE.format(accent=accent)),
        ])
        qspec.limit = min(qspec.limit or batch_limit, batch_limit)

        for it in range(max_iters):
            need = target_n - len(approved_ids)
            if need <= 0:
                break

            # 1) Тянем новый batch из БД по текущему qspec
            pool = get_from_query(qspec, session=session, top_n=qspec.limit or batch_limit)
            # выбросим уже виденных
            pool = [c for c in pool if c.id not in seen_ids]

            if not pool:
                # 2a) Если новых нет — просим LLM расширить запрос
                relax = llm.with_structured_output(QuerySpec).invoke([
                    SystemMessage(content=RELAX_QUERY_SYSTEM_PROMPT),
                    HumanMessage(content=(
                        f"Акцент: {accent}\n"
                        f"Нужно добрать ещё: {need}\n"
                        f"Текущая спецификация: {qspec.model_dump_json()}\n"
                        f"Уже виденные id (исключи их): {[str(i) for i in seen_ids]}"
                    ))
                ])
                relax.limit = batch_limit
                qspec = relax
                continue

            # 2b) Просим LLM выбрать до need id из pool
            norm_llm = llm.with_structured_output(NormIDs)
            candidates_for_llm = [{
                "id": str(c.id),
                "desired_position": c.desired_position,
                "city": c.city,
                "expected_salary_rub": c.expected_salary_rub,
                "resume_updated_at": c.resume_updated_at.isoformat() if c.resume_updated_at else None,
                "work_experience": _short(c.work_experience),
            } for c in pool]

            sys = SystemMessage(content=(
                f"Выбери ДО {need} наиболее релевантных кандидатов под исходный запрос. "
                "Верни строго JSON модели NormIDs: {\"candidates\": [<uuid>, ...]}. "
                "Используй только id из предоставленного списка, без дублей."
            ))
            user = HumanMessage(content=(
                f"Исходный запрос: {state.get('task','')}\n"
                f"Уточнение: {state.get('extra_task','')}\n\n"
                f"Кандидаты:\n{json.dumps(candidates_for_llm, ensure_ascii=False)}"
            ))

            try:
                picked = norm_llm.invoke([sys, user]).candidates
            except Exception:
                picked = []

            # фильтруем только новых, которых ещё не брали
            picked = [pid for pid in picked if pid not in approved_ids and pid not in seen_ids]

            # 3) Добавляем выбранных, помечаем approved=True
            picked_set = set(picked)
            for c in pool:
                seen_ids.add(c.id)
                if c.id in picked_set and len(approved_ids) < target_n:
                    group.candidates.append(CandidateScore(candidate=c, approved=True))
                    approved_ids.add(c.id)

            # 4) Если всё ещё не хватает — просим LLM ослабить запрос и идём на следующую итерацию
            if len(approved_ids) < target_n:
                relax = llm.with_structured_output(QuerySpec).invoke([
                    SystemMessage(content=RELAX_QUERY_SYSTEM_PROMPT),
                    HumanMessage(content=(
                        f"Акцент: {accent}\n"
                        f"Нужно добрать ещё: {target_n - len(approved_ids)}\n"
                        f"Текущая спецификация: {qspec.model_dump_json()}\n"
                        f"Уже отобранные id: {[str(i) for i in approved_ids]}\n"
                        f"Уже виденные id (исключи их): {[str(i) for i in seen_ids]}"
                    ))
                ])
                relax.limit = batch_limit
                qspec = relax

    return state

def node_rate_candidates(state: State, llm, top_n: int = 10) -> State:
    """
    Ранжирует кандидатов силами LLM:
    - собирает плоский список кандидатов из state["raw_candidates"];
    - просит модель вернуть NormIDs (до top_n id в нужном порядке);
    - формирует state["ranked"] как список CandidateScore в этом порядке.
    Если что-то пошло не так — фолбэк: approved сначала, затем прочие.
    """
    # 1) Дедуп по id и объединение признака approved (OR между группами/акцентами).
    items_by_id: dict[uuid.UUID, dict] = {}
    for group in state.get("raw_candidates", []):
        for cs in group.candidates:
            cid = cs.candidate.id
            if cid not in items_by_id:
                items_by_id[cid] = {"candidate": cs.candidate, "approved": bool(cs.approved)}
            else:
                items_by_id[cid]["approved"] = items_by_id[cid]["approved"] or bool(cs.approved)

    if not items_by_id:
        state["ranked"] = []
        return state

    # 2) Подготовка компактного JSON для LLM (не раздуваем контекст).
    def _short(txt: Optional[str], limit: int = 600) -> Optional[str]:
        if not txt:
            return txt
        return txt if len(txt) <= limit else (txt[:limit] + "…")

    candidates_for_llm = [{
        "id": str(cid),
        "approved": item["approved"],
        "desired_position": item["candidate"].desired_position,
        "city": item["candidate"].city,
        "expected_salary_rub": item["candidate"].expected_salary_rub,
        "resume_updated_at": item["candidate"].resume_updated_at.isoformat() if item["candidate"].resume_updated_at else None,
        "work_experience": _short(item["candidate"].work_experience),
    } for cid, item in items_by_id.items()]

    # 3) Просим LLM вернуть строго NormIDs (до top_n uuid'ов, только из данного списка).
    sys = SystemMessage(content=(
        "Ты — ассистент рекрутера. По списку кандидатов выбери ДО 10 самых релевантных под исходный запрос. "
        "Сильный сигнал — approved=True; также учитывай позицию, город, зарплату, краткое описание опыта. "
        f"Верни строго JSON модели NormIDs: {{ \"candidates\": [<uuid>, ...] }}, максимум {top_n} штук, "
        "только из переданных id, без придуманных значений и без дублей."
    ))
    user = HumanMessage(content=(
        f"Исходный запрос: {state.get('task','')}\n"
        f"Уточнение: {state.get('extra_task','')}\n\n"
        f"Кандидаты:\n{json.dumps(candidates_for_llm, ensure_ascii=False)}"
    ))

    try:
        structured = llm.with_structured_output(NormIDs)
        out: NormIDs = structured.invoke([sys, user])

        # 4) Нормализуем: оставляем только существующие id, сохраняем порядок, режем до top_n.
        allowed = {str(k) for k in items_by_id.keys()}
        seen: set[str] = set()
        ordered_ids: list[str] = []
        for cid in map(str, out.candidates):
            if cid in allowed and cid not in seen:
                ordered_ids.append(cid)
                seen.add(cid)
            if len(ordered_ids) >= top_n:
                break

        # 5) Собираем итоговый ranked: выбранных LLM помечаем approved=True.
        ranked: List[CandidateScore] = []
        for s_id in ordered_ids:
            cid = uuid.UUID(s_id)
            item = items_by_id[cid]
            ranked.append(CandidateScore(candidate=item["candidate"], approved=True))

        # Если LLM вернула меньше top_n — дозаполним оставшимися (approved сначала).
        if len(ranked) < top_n:
            remaining = [cid for cid in items_by_id.keys() if str(cid) not in seen]
            remaining.sort(
                key=lambda cid: (
                    not items_by_id[cid]["approved"],
                    - (items_by_id[cid]["candidate"].expected_salary_rub or 0)  # вторичный ключ, можно заменить
                )
            )
            for cid in remaining:
                if len(ranked) >= top_n:
                    break
                item = items_by_id[cid]
                ranked.append(CandidateScore(candidate=item["candidate"], approved=item["approved"]))

        state["ranked"] = ranked
        return state

    except Exception:
        # Фолбэк: без LLM — approved сначала, затем остальные (до top_n).
        ordered = sorted(
            items_by_id.values(),
            key=lambda it: (not it["approved"], - (it["candidate"].expected_salary_rub or 0))
        )[:top_n]
        state["ranked"] = [CandidateScore(candidate=it["candidate"], approved=it["approved"]) for it in ordered]
        return state

def node_ask_next(state: State) -> State:
    try:
        followup = input(ASK_NEXT_PROMPT)
    except EOFError:
        followup = ""
    if followup.strip():
        state["extra_task"] = followup.strip()
    return state
