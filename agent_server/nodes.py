from typing import Dict, List, Callable
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from candidates import Candidate, CandidateScore, TopCandidates

class QueryVariants(BaseModel):
    accents: List[str] = Field(
        ...,
        description="Разные формулировки запроса/акценты, по которым можно искать кандидатов",
    )

def node_get_task(state: Dict) -> Dict:
    state["task"] = input("Кого будем искать? ")
    state["extra_task"] = ""
    return state

def node_generate_accents(state: State, llm) -> State:
    msgs = [
        SystemMessage(
            content=(
                "Тебе дают запрос на кандидата. "
                "Сформулируй 3-5 разных поисковых акцентов. "
                "Каждый акцент — отдельная формулировка профиля, по которой можно искать."
            )
        ),
        HumanMessage(content=state["task"]),
    ]

    if state.get("ranked"):
        msgs.append(
            SystemMessage(
                content="Вот прошлые лучшие кандидаты и уточнение рекрутера:"
            )
        )
        msgs.append(AIMessage(content=str(state["ranked"])))
        if state.get("extra_task"):
            msgs.append(HumanMessage(content=state["extra_task"]))

    accents_resp = structured_llm.invoke(msgs)
    state["accents"] = accents_resp.accents
    return state

class ToolUse(BaseModel):
    query: str
    intent: str

class RankedCandidateRef(BaseModel):
    candidate_id: str
    score: float = Field(..., ge=0, le=10)
    reason: str

class StepAnswer(BaseModel):
    ranked: List[RankedCandidateRef] = Field(
        ...,
        description="Финальный топ кандидатов по этому акценту (максимум 5), отсортированный по убыванию score"
    )

class StepEnvelope(BaseModel):
    step: Union[ToolStep, StepAnswer] = Field(
        ...,
        description=(
            "Либо {'kind':'tool','tool_use':{...}}, "
            "либо {'kind':'answer','answer':{'ranked':[...]}}, но только один из вариантов."
        )
    )

def node_choose_candidates(state: State, llm, tools_node, max_tool_rounds = 5) -> State:
    structured_llm = llm_with_tools.with_structured_output(StepEnvelope)

    def _inner(state: Dict) -> Dict:
        accents: List[str] = state.get("accents", [])
        result_groups: List[TopCandidates] = []

        for accent in accents:
            history_rows: List[Dict] = []
            final_rank_refs: List[RankedCandidateRef] = []
            max_rounds = 5

            for _ in range(max_rounds):
                sys_msg = SystemMessage(
                    content=(
                        "Ты помогаешь рекрутеру находить подходящих кандидатов по базе данных candidates.\n\n"
                        "У тебя нет права придумывать содержимое резюме — факты берем только из БД.\n\n"
                        "На каждом шаге ТЫ ДОЛЖЕН вернуть JSON с полем 'step'.\n"
                        "Есть два допустимых формата 'step':\n\n"
                        "1) Шаг запроса к базе (kind='tool'):\n"
                        "{\n"
                        "  'kind': 'tool',\n"
                        "  'tool_use': {\n"
                        "     'query': 'SELECT id, desired_position, city, work_experience, "
                        "expected_salary_rub FROM candidates WHERE ... LIMIT 5',\n"
                        "     'intent': 'что ты хочешь получить и почему это нужно для данного акцента'\n"
                        "  }\n"
                        "}\n\n"
                        "Правила для kind='tool':\n"
                        "- Только SELECT.\n"
                        "- Обязательно LIMIT <=5.\n"
                        "- Возвращай только важные колонки (id, desired_position, city, work_experience, expected_salary_rub, ...).\n"
                        "- Используй условия WHERE, чтобы сужать выборку под акцент.\n\n"
                        "2) Финальный ответ (kind='answer'):\n"
                        "{\n"
                        "  'kind': 'answer',\n"
                        "  'answer': {\n"
                        "     'ranked': [\n"
                        "        {\n"
                        "          'candidate_id': '<ID кандидата из БД>',\n"
                        "          'score': 8.7,\n"
                        "          'reason': 'почему кандидат релевантен под задачу',\n"
                        "        },\n"
                        "        ... максимум 5 штук ...\n"
                        "     ]\n"
                        "  }\n"
                        "}\n\n"
                        "Правила для kind='answer':\n"
                        "- НЕ возвращай факты кандидата (город, зарплата и т.д.) как истину. "
                        "Только candidate_id, score и reason.\n"
                        "- Используй только те candidate_id, которые ты реально видел в rows из БД.\n"
                        "- score от 0 до 10, где 10 = максимально релевантен запросу.\n"
                        "- reason — краткое объяснение логики выбора.\n\n"
                        "НИКОГДА не смешивай оба шага одновременно. Либо 'tool', либо 'answer'."
                    )
                )

                human_msg = HumanMessage(
                    content=(
                        "Поисковый акцент:\n"
                        f"{accent}\n\n"
                        "Вот данные, которые у тебя уже есть из БД на предыдущих шагах "
                        "(каждый элемент — результат одного SQL):\n"
                        f"{json.dumps(history_rows, ensure_ascii=False)}\n\n"
                        "Сейчас верни ЛИБО шаг kind='tool' с нужным SELECT, "
                        "ЛИБО шаг kind='answer' с финальным ranked."
                    )
                )

                envelope: StepEnvelope = structured_llm.invoke([sys_msg, human_msg])
                step = envelope.step

                if isinstance(step, ToolStep):
                    tool_result = query_tool.run({"query": step.tool_use.query})
                    history_rows.append({
                        "query": tool_result["query"],
                        "error": tool_result["error"],
                        "rows": tool_result["rows"],
                    })

                else:
                    final_rank_refs = step.answer.ranked
                    break

            ranked_ids = [ref.candidate_id for ref in final_rank_refs]
            factual_by_id = fetch_candidates_by_ids(engine, ranked_ids)

            scored_list: List[CandidateScore] = []

            for ref in final_rank_refs:
                factual = factual_by_id.get(str(ref.candidate_id))
                cand = Candidate(
                    id=factual["id"],
                    sex=factual["sex"],
                    expected_salary_rub=factual["expected_salary_rub"],
                    desired_position=factual["desired_position"],
                    city=factual["city"],
                    ready_to_relocate=factual["ready_to_relocate"],
                    ready_for_business_trips=factual["ready_for_business_trips"],
                    employment_type=factual["employment_type"],
                    work_schedule=factual["work_schedule"],
                    work_experience=factual["work_experience"],
                    last_company=factual["last_company"],
                    last_job_title=factual["last_job_title"],
                    education_level_and_university=factual["education_level_and_university"],
                    resume_updated_at=factual["resume_updated_at"],
                    has_car=factual["has_car"],
                )

                scored_list.append(
                    CandidateScore(
                        candidate=cand,
                        score=ref.score,
                        reason=ref.reason,
                    )
                )
            all_top_groups = state["raw_candidates"]
            all_top_groups.append(
                TopCandidates(
                    accent=accent,
                    candidates=scored_list,
                )
            )
        state["raw_candidates"] = all_top_groups

        return state

    return _inner

def node_rate_candidates(state: State, llm) -> State:
        flat_candidates: List[Candidate] = []
        for group in state.get("raw_candidates", []):
            for cs in group.candidates:
                flat_candidates.append(cs.candidate)

        structured_llm = llm.with_structured_output(List[CandidateScore])  # типовая идея

        msgs = [
            SystemMessage(
                content=(
                    "Тебе дан список кандидатов (с краткой инфой по каждому) "
                    "и запрос. Оцени каждого по шкале 0-10 и верни топ-10."
                )
            ),
            HumanMessage(content=f"Запрос: {state['task']}"),
            HumanMessage(
                content="\n".join(c.model_dump_json() for c in flat_candidates)
            ),
        ]

        ranked: List[CandidateScore] = structured_llm.invoke(msgs)

        state["ranked"] = sorted(
            ranked, key=lambda x: x.score, reverse=True
        )[:10]

        return state

def node_return_candidates(state: State) -> State:
        top_ids = [c.candidate.id for c in state.get("ranked", [])]
        full_objects = fetch_candidates_by_ids(top_ids)

        print("\nНашли таких кандидатов:")
        for cand in full_objects:
            print(
                f"- {cand.id}: {cand.desired_position}, {cand.city}, "
                f"зарплата {cand.expected_salary_rub}, опыт: {cand.work_experience}"
            )

        return state

def node_ask_next(state: Dict) -> Dict:
    followup = input(
        "\nДобавьте уточнение (например, 'нужен опыт с Django' или просто Enter, чтобы закончить): "
    )

    state["extra_task"] = followup.strip()
    return state