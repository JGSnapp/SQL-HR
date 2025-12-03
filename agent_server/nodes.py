from __future__ import annotations

import os
import uuid
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Literal

logger = logging.getLogger(__name__)
RESULT_FILE = Path(
    os.getenv(
        "RESULT_FILE",
        Path(__file__).resolve().parent / "result.txt",
    )
)

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
from prompts import CHOOSE_CANDIDATES_SYSTEM_PROMPT

RELAX_QUERY_SYSTEM_PROMPT = (
    "Тебе дан акцент (формулировка поиска) и текущая выборка кандидатов. "
    "Нужно ДОБРАТЬ ещё кандидатов: сгенерируй новую спецификацию QuerySpec, которая расширит охват, "
    "но останется релевантной: допускается смягчить salary, добавить синонимы в keywords_any, "
    "снять часть keywords_all, расширить город (агломерации/удалёнка). "
    "Всегда отвечай по-русски, кроме названий/имен. "
    "Всегда указывай limit (batch_limit). Верни строго JSON модели QuerySpec, никакого текста вне JSON."
)

# --- Простые текстовые промпты по умолчанию (можно заменить своими из prompts.py) ---
GET_TASK_PROMPT = (
    "Найди опытного веб-разработчика (Fullstack) на WordPress с сильным знанием "
    "HTML, CSS, JavaScript и PHP. Обязателен опыт более 10 лет, работа с CMS, "
    "интеграциями и поддержкой крупных проектов."
)
GENERATE_ACCENTS_SYSTEM_PROMPT = (
    "Ты генерируешь 2–5 разных формулировок запроса ('акцентов') для поиска кандидатов. "
    "Коротко, по сути. Верни JSON со списком 'accents'."
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
    city: Optional[str] = Field(
        default=None,
        description="Название города/региона, который должен упоминаться в поле `city` кандидата.",
    )
    min_salary_rub: Optional[int] = Field(
        default=None,
        description="Нижняя граница ожидаемой зарплаты кандидата (в рублях).",
    )
    max_salary_rub: Optional[int] = Field(
        default=None,
        description="Верхняя граница ожидаемой зарплаты кандидата (в рублях).",
    )
    ready_to_relocate: Optional[bool] = Field(
        default=None,
        description="True/False, требуется ли готовность переезда.",
    )
    keywords_any: List[str] = Field(
        default_factory=list,
        description="Список ключевых слов, из которых должно встретиться хотя бы одно в `work_experience`.",
    )
    keywords_all: List[str] = Field(
        default_factory=list,
        description="Ключевые слова, которые обязательно должны присутствовать одновременно в `work_experience`.",
    )
    keywords_not: List[str] = Field(
        default_factory=list,
        description="Ключевые слова, которые не должны встречаться в `work_experience`.",
    )
    seniority: Optional[Literal["junior", "middle", "senior", "lead"]] = Field(
        default=None,
        description="Требуемый уровень кандидата (одно из: junior/middle/senior/lead).",
    )
    limit: int = Field(
        5,
        ge=5,
        le=100,
        description="Сколько строк вернуть в рамках одного запроса к БД.",
    )


class QueryVariants(BaseModel):
    accents: List[str] = Field(..., min_items=1, max_items=8)

# --- Поиск через ORM по собранному QuerySpec ---
def get_from_query(spec: QuerySpec, session: Session, top_n: int = 5) -> List[CandidateOut]:
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

def node_generate_accents(state: State, llm) -> dict:
    msgs = [
        SystemMessage(content=GENERATE_ACCENTS_SYSTEM_PROMPT),
        HumanMessage(content=state.get("task", "")),
    ]
    structured = llm.with_structured_output(QueryVariants)
    accents_resp = structured.invoke(msgs)
    logger.info("Вывод LLM в node_generate_accents: %s", accents_resp.model_dump())
    result = {"accents": accents_resp.accents}
    logger.info("Обновленные данные после node_generate_accents: %s", result)
    return result

def node_choose_candidates(state: State, llm, session: Session) -> dict:
    # Копируем уже существующее, если есть
    groups: List[TopCandidates] = list(state.get("raw_candidates", []))

    def _short(txt: Optional[str], limit: int = 500) -> Optional[str]:
        if not txt:
            return txt
        return txt if len(txt) <= limit else (txt[:limit] + "…")

    for accent in state.get("accents", []):
        # 1) LLM превращает акцент в QuerySpec
        qspec_llm = llm.with_structured_output(QuerySpec)
        llm_query = qspec_llm.invoke([
            SystemMessage(content=CHOOSE_CANDIDATES_SYSTEM_PROMPT),
            HumanMessage(content=CHOOSE_CANDIDATES_HUMAN_PROMPT_TEMPLATE.format(accent=accent)),
        ])

        logger.info("Вывод LLM в node_choose_candidates (QuerySpec): %s", llm_query.model_dump())

        # 2) ORM-поиск по спецификации (жестко ограничиваем выборку 5 строками)
        candidates = get_from_query(llm_query, session=session, top_n=5)

        candidates_for_llm = [{
            "id": str(c.id),
            "desired_position": c.desired_position,
            "city": c.city,
            "expected_salary_rub": c.expected_salary_rub,
            "resume_updated_at": c.resume_updated_at.isoformat() if c.resume_updated_at else None,
            "work_experience": _short(c.work_experience),
        } for c in candidates]

        # 3) Нормализация/отбор id через LLM
        norm_llm = llm.with_structured_output(NormIDs)
        msgs = [
            SystemMessage(content=RATE_CANDIDATES_SYSTEM_PROMPT),
            HumanMessage(content=f"Запрос: {state.get('task','')}"),
            HumanMessage(content=f"Кандидаты:\n{json.dumps(candidates_for_llm, ensure_ascii=False)}"),
        ]
        try:
            norm_ids = norm_llm.invoke(msgs).candidates
            logger.info("Вывод LLM в node_choose_candidates (NormIDs): %s", [str(nid) for nid in norm_ids])
            ids_set = set(norm_ids)
            ready = [CandidateScore(candidate=c, approved=(c.id in ids_set)) for c in candidates]
        except Exception:
            logger.exception(
                "node_choose_candidates: failed to score candidates for accent `%s`, approving all",
                accent,
            )
            ready = [CandidateScore(candidate=c, approved=True) for c in candidates]

        groups.append(TopCandidates(accent=accent, candidates=ready))

    logger.info("Обновленные данные после node_choose_candidates: %s", {"groups": len(groups)})
    return {"raw_candidates": groups}


def node_test_db(state: State, session: Session) -> dict:
    sample = session.execute(select(C).limit(5)).scalars().all()
    serialized = [CandidateOut.model_validate(item).model_dump() for item in sample]
    logger.info("Выборка test_db (5 записей): %s", serialized)
    return {}


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
    target_n: int = 5,       # сколько в итоге хотим на 1 акцент
    batch_limit: int = 5,    # сколько тянуть за итерацию из БД
    max_iters: int = 3,      # максимум итераций расширения на акцент
) -> dict:
    """
    Для каждого акцента, уже имеющегося в state['raw_candidates'],
    добирает недостающих кандидатов: на каждой итерации LLM ослабляет QuerySpec,
    мы тянем batch из БД и просим LLM выбрать до недостающего количества id.
    """

    # helper: аккуратно урезать текст, чтобы не раздувать контекст LLM
    def _short(txt: Optional[str], limit: int = 500) -> Optional[str]:
        if not txt:
            return txt
        return txt if len(txt) <= limit else (txt[:limit] + "…")
    
    raw_candidates = list(state.get("raw_candidates", []))

    for group in raw_candidates:
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
        logger.info(
            "Вывод LLM в node_add_candidates (base QuerySpec): %s",
            {"accent": accent, "qspec": qspec.model_dump()},
        )
        qspec.limit = min(qspec.limit or batch_limit, batch_limit)

        for it in range(max_iters):
            need = target_n - len(approved_ids)
            if need <= 0:
                break

            # 1) Тянем новый batch из БД по текущему qspec
            pool = get_from_query(qspec, session=session, top_n=5)
            # выбросим уже виденных
            pool = [c for c in pool if c.id not in seen_ids]

            if not pool:
                # 2a) Если новых нет — просим LLM расширить запрос
                try:
                    relax = llm.with_structured_output(QuerySpec).invoke([
                        SystemMessage(content=RELAX_QUERY_SYSTEM_PROMPT),
                        HumanMessage(content=(
                            f"Акцент: {accent}\n"
                            f"Нужно добрать ещё: {need}\n"
                            f"Текущая спецификация: {qspec.model_dump_json()}"
                        ))
                    ])
                    relax.limit = 5
                    qspec = relax
                except Exception:
                    logger.exception("node_add_candidates: не удалось ослабить запрос для акцента `%s`", accent)
                    break
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
                logger.info("Вывод LLM в node_add_candidates (picked ids): %s", {"accent": accent, "iteration": it + 1, "picked": [str(pid) for pid in picked]})
            except Exception:
                logger.exception(
                    "node_add_candidates: LLM failed to pick candidates for accent `%s` iteration %d",
                    accent,
                    it + 1,
                )
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
                relax.limit = 5
                qspec = relax

    logger.info("Обновленные данные после node_add_candidates: %s", {"groups_processed": len(raw_candidates)})
    return {}

def _dedupe_scores(items: List[CandidateScore]) -> List[CandidateScore]:
    """Remove duplicated candidates while preserving ordering."""
    seen: set[uuid.UUID] = set()
    deduped: list[CandidateScore] = []
    for item in items:
        cid = item.candidate.id
        if cid in seen:
            continue
        seen.add(cid)
        deduped.append(item)
    return deduped


def node_rate_candidates(state: State, llm, top_n: int = 5) -> dict:
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
        logger.info("node_rate_candidates: nothing to rank")
        return {"ranked": []}

    # 2) Подготовка компактного JSON для LLM (не раздуваем контекст).
    def _short(txt: Optional[str], limit: int = 500) -> Optional[str]:
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

        ranked = _dedupe_scores(ranked)
        logger.info("node_rate_candidates: ranked %d candidates", len(ranked))
        return {"ranked": ranked}

    except Exception:
        # Фолбэк: без LLM — approved сначала, затем остальные (до top_n).
        logger.exception("node_rate_candidates: exception during ranking, falling back to deterministic ordering")
        ordered = sorted(
            items_by_id.values(),
            key=lambda it: (not it["approved"], - (it["candidate"].expected_salary_rub or 0))
        )[:top_n]
        fallback_ranked = _dedupe_scores([
            CandidateScore(candidate=it["candidate"], approved=it["approved"])
            for it in ordered
        ])
        logger.info("node_rate_candidates: fallback ranked %d candidates", len(fallback_ranked))
        return {"ranked": fallback_ranked}

def node_return_candidates(state: State) -> dict:
    ranked = state.get("ranked")
    if not ranked:
        logger.info("node_return_candidates: nothing to return")
        return {}
    
    def _short(txt: Optional[str], limit: int = 500) -> Optional[str]:
        if not txt:
            return txt
        return txt if len(txt) <= limit else (txt[:limit] + "…")
    
    logger.info("node_return_candidates: deduped to %d entries", len(ranked))
    serialized = []
    for item in ranked:
        candidate = item.candidate
        serialized.append({
            "id": str(candidate.id),
            "approved": bool(item.approved),
            "desired_position": candidate.desired_position,
            "city": candidate.city,
            "expected_salary_rub": candidate.expected_salary_rub,
            "ready_to_relocate": candidate.ready_to_relocate,
            "resume_updated_at": candidate.resume_updated_at.isoformat()
            if candidate.resume_updated_at else None,
            "work_experience": _short(candidate.work_experience),
        })

    payload = {"total": len(serialized), "candidates": serialized}
    targets = [RESULT_FILE]
    fallback = Path(tempfile.gettempdir()) / RESULT_FILE.name
    if fallback not in targets:
        targets.append(fallback)

    dump = json.dumps(payload, ensure_ascii=False, indent=2)
    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(dump, encoding="utf-8")
            logger.info("node_return_candidates: wrote results to %s", target)
            break
        except PermissionError:
            logger.warning(
                "node_return_candidates: permission denied for %s, trying fallback",
                target,
            )
        except Exception:
            logger.exception(
                "node_return_candidates: failed to write results to %s",
                target,
            )
    else:
        logger.error(
            "node_return_candidates: exhausted all targets %s",
            [str(t) for t in targets],
        )

    return {}

def node_ask_next(state: State) -> dict:
    followup = ASK_NEXT_PROMPT
    logger.info("node_ask_next: recorded followup `%s`", followup[:80])
    return {"extra_task": followup}

