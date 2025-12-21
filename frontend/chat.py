import streamlit as st
import requests
import os

SERVER_URL = os.getenv("SERVER_URL")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "candidates" not in st.session_state:
    st.session_state.candidates = {}

if "session_id" not in st.session_state:
    st.session_state.session_id = None

st.set_page_config(page_title="SQL-HR", layout="wide")

st.title("SQL-HR")


# Две равные колонки
left, right = st.columns(2)

with left:
    st.subheader("Чат")

    # История сообщений
    for m in st.session_state.messages:
        content = m["content"]
        if m["role"] == "assistant" and "tool_call" in content:
            content = "Ищу по базе..."
        with st.chat_message(m["role"]):
            st.markdown(content)

# Инпут всегда внизу окна (особенность st.chat_input)
prompt = st.chat_input("Введите запрос")

if prompt:
    # Добавляем сообщение пользователя в историю
    st.session_state.messages.append({"role": "user", "content": prompt})

    reply = ""
    candidates = {}

    if not SERVER_URL:
        reply = "SERVER_URL не задан в переменных окружения."
    else:
        with st.spinner("Отправляю запрос..."):
            try:
                payload = {
                    "session_id": st.session_state.session_id,
                    "message": prompt,
                }
                r = requests.post(
                    SERVER_URL,
                    json=payload,
                )
                r.raise_for_status()
                r_json = r.json()

                # обновляем session_id от сервера
                st.session_state.session_id = r_json.get(
                    "session_id", st.session_state.session_id
                )

                reply = r_json.get("answer", "")
                # сервер отдаёт словарь: id -> данные кандидата
                candidates = r_json.get("candidates", {}) or {}
            except Exception as e:
                reply = f"Ошибка при запросе к серверу: {e}"
                candidates = {}

    # сохраняем кандидатов в состоянии
    st.session_state.candidates = candidates

    # добавляем ответ ассистента в историю
    if reply:
        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )

    # перерисовываем, чтобы увидеть новое сообщение
    st.rerun()

with right:
    st.subheader("Кандидаты")

    if st.session_state.candidates:
        # st.session_state.candidates — dict: id -> dict(поля кандидата)
        for cand_id, candidate in st.session_state.candidates.items():
            with st.container(border=True):
                # заголовок карточки = desired_position
                desired = candidate.get("desired_position", "Позиция не указана")
                st.markdown(f"#### {desired}")

                # видно сразу
                st.markdown(f"**ID:** `{cand_id}`")
                if "expected_salary_rub" in candidate:
                    st.markdown(
                        f"**Ожидаемая зарплата, ₽:** {candidate['expected_salary_rub']}"
                    )

                # остальные поля под стрелкой вниз
                with st.expander("Показать остальные поля"):
                    for field, value in candidate.items():
                        if field in ("desired_position", "expected_salary_rub"):
                            continue
                        st.markdown(f"**{field}:** {value}")
    else:
        st.caption("Пока кандидатов нет")
