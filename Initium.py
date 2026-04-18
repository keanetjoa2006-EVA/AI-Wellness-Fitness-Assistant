import streamlit as st
import json
from Initium_Brain import ask_gym_ai, needs_more_context, ask_for_context, form_answer_with_context
from Initium_Data_Storage import get_gym_data

gym_collection = get_gym_data()

st.title("Initium")
st.caption("")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "waiting_for_context" not in st.session_state:
    st.session_state.waiting_for_context = False

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
if prompt := st.chat_input("Ask me anything about fitness..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    status = st.empty()
    assistant_avatar = "🤖"

    if st.session_state.waiting_for_context:
        with status.spinner("Thinking..."):
            full_response = ""
            response = form_answer_with_context(st.session_state.pending_question, prompt)
            if isinstance(response, str):
                full_response = response
            else:
                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("done", False):
                        break
                    token = data.get("response") or data.get("text") or data.get("token") or ""
                    if token:
                        full_response += token
        status.empty()

        with st.chat_message("assistant", avatar=assistant_avatar):
            st.write(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.waiting_for_context = False
        st.session_state.pending_question = None

    elif needs_more_context(prompt):
        with status.spinner("Thinking..."):
            clarification = ask_for_context(prompt)
        status.empty()

        with st.chat_message("assistant", avatar=assistant_avatar):
            st.write(clarification)
        st.session_state.messages.append({"role": "assistant", "content": clarification})
        st.session_state.waiting_for_context = True
        st.session_state.pending_question = prompt

    else:
        with status.spinner("Thinking..."):
            full_response = ""
            response = ask_gym_ai(prompt, stream=True)
            if isinstance(response, str):
                full_response = response
            else:
                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("done", False):
                        break
                    token = data.get("response") or data.get("text") or data.get("token") or ""
                    if token:
                        full_response += token
        status.empty()

        if not full_response:
            full_response = "Sorry, I could not retrieve an answer."
        with st.chat_message("assistant", avatar=assistant_avatar):
            st.write(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
