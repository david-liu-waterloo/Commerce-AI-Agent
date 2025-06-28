import streamlit as st

st.title("Mock Commercial Website AI Agent")

# to memorize all existing messages
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chat input
if prompt := st.chat_input(placeholder="Say something to our AI Agent...",
                    accept_file=True,
                    file_type=[".png", ".jpg", ".jpeg"]):
    # prompt object: (text="{string}", files=[])

    # user query
    if prompt.text:
        st.chat_message("role").markdown(prompt.text)
        st.session_state.messages.append({"role": "user", "content": prompt.text})
    if prompt["files"]:
        st.chat_message("role").image(prompt["files"][0])

    # AI response
    response = f"{prompt.text} (TODO: GET ACTUAL AI RESPONSE)" # TODO: get response from AI in ./app.py

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})