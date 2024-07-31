import streamlit as st
import os

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.storage import LocalFileStore
from langchain.memory import ConversationSummaryBufferMemory

from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Streamlit Page Configuration
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

# Ensure the directories exist
os.makedirs("./.cache/files/", exist_ok=True)
os.makedirs("./.cache/embeddings/", exist_ok=True)

# API Key Inputs
openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key:", type="password")
langchain_api_key = st.sidebar.text_input(
    "Enter your LangChain API Key:", type="password")

if openai_api_key and langchain_api_key:
    # Set LangChain API key as environment variable
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

    # Define Classes and Functions

    # Chat Callback Handler Class
    class ChatCallbackHandler(BaseCallbackHandler):
        message = ""

        def on_llm_start(self, *args, **kwargs):
            self.message_box = st.empty()

        def on_llm_end(self, *args, **kwargs):
            save_message(self.message, "ai")

        def on_llm_new_token(self, token, *args, **kwargs):
            self.message += token
            self.message_box.markdown(self.message)

    if "memory" not in st.session_state:
        st.session_state.memory = ChatCallbackHandler()

    # LLM Initialization
    def get_llm(api_key):
        return ChatOpenAI(
            api_key=api_key,
            temperature=0.1,
            streaming=True,
            callbacks=[
                ChatCallbackHandler(),
            ],
        )

    # Memory Initialization
    @st.cache_resource
    def init_memory(_llm_for_memory):
        return ConversationSummaryBufferMemory(
            llm=_llm_for_memory,
            max_token_limit=120,
            memory_key="history",
            return_messages=True
        )

    # Embed File
    @st.cache_resource(show_spinner="Embedding file...")
    def embed_file(file, openai_api_key):
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_dir)
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        retriever = vectorstore.as_retriever()
        return retriever

    # Save Message
    def save_message(message, role):
        st.session_state["messages"].append({"message": message, "role": role})

    # Send Message
    def send_message(message, role, save=True):
        with st.chat_message(role):
            st.markdown(message)
        if save:
            save_message(message, role)

    # Paint History
    def paint_history():
        for message in st.session_state["messages"]:
            send_message(
                message["message"],
                message["role"],
                save=False,
            )

    # Format Documents
    def format_docs(docs):
        return "\n\n".join(document.page_content for document in docs)

    # Load Memory
    def load_memory(_):
        return memory.load_memory_variables({})["history"]

    # Clear Memory
    def clear_memory():
        st.session_state["messages"] = []
        memory.clear()

    # Initialize LLM and Memory
    llm = get_llm(openai_api_key)
    llm_for_memory = get_llm(openai_api_key)
    memory = init_memory(llm_for_memory)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
                
                Context: {context}
                """,
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # Streamlit App Title and Description
    st.title("DocumentGPT")

    st.markdown(
        """
    Welcome!
                
    Use this chatbot to ask questions to an AI about your files!

    Upload your files on the sidebar.
    """
    )

    # Sidebar File Uploader
    with st.sidebar:
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],
            on_change=clear_memory,
        )

    # File Handling and Chat Interface
    if file:
        retriever = embed_file(file, openai_api_key)
        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file...")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "history": RunnableLambda(load_memory),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                response = chain.invoke(message)

            memory.save_context({"input": message}, {
                                "output": response.content})

    # Initialize the session if user changes the file
    else:
        st.session_state["messages"] = []
        clear_memory()
else:
    st.warning(
        "Please enter your OpenAI API key and LangChain API key in the sidebar to proceed.")
