import os
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler

# To store cache data, make sure the folder exist


def make_dir():
    os.makedirs("./.cache/files", exist_ok=True)
    os.makedirs("./.cache/embeddings", exist_ok=True)
    os.makedirs("./.cache/chunks", exist_ok=True)
    os.makedirs("./.cache/history", exist_ok=True)
    os.makedirs("./.cache/private_embeddings", exist_ok=True)
    os.makedirs("./.cache/private_files", exist_ok=True)
    os.makedirs("./.cache/quiz_files", exist_ok=True)


def save_message(msg, role):
    st.session_state["messages"].append({"msg": msg, "role": role})


def send_message(msg, role, save=True):
    with st.chat_message(role):
        st.markdown(msg)
    if save:
        save_message(msg, role)


def paint_history():
    for msg in st.session_state["messages"]:
        send_message(msg["msg"], msg["role"], False)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)
