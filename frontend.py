import streamlit as st
import app

st.title("Mock Commercial Website AI Agent")

# to display all previous messages
# TODO: display images for AI responses
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        match message["role"]:

            case "assistant":
                st.markdown(message["content"])

            case "user":
                st.markdown(message["content"].text)
                if message["content"].files:
                    st.image(message["content"].files[0])

# chat input
if prompt := st.chat_input(placeholder="Say something to our AI Agent...",
                    max_chars=2000, # I do not have infinite budget for tokens
                    accept_file=True,
                    file_type=[".png", ".jpg", ".jpeg"]):
    # prompt object: (text="{string}", files=[])

    # user query
    if prompt.text:
        with st.chat_message("user"):
            st.markdown(prompt.text)
            if prompt["files"]:
                st.image(prompt["files"][0])
            st.session_state.messages.append({"role": "user", "content": prompt})

    # AI response
    response = app.get_response(prompt.text)

    # DEBUG
    print("[DEBUG] \n", response)

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})