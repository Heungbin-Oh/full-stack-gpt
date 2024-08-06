from langchain.storage import LocalFileStore
import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
# utils
from utils.utils import paint_history, send_message
from utils.chat_handler import get_answers_no_src, choose_answer_no_src, memory
from utils.data_loader import find_history, load_memory


llm = ChatOpenAI(
    temperature=0.1,
)


has_transcript = os.path.exists("./.cache/podcast.txt")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data()
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            text_file.write(transcript["text"])


@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(
            f"./{chunks_folder}/chunk_{i}.mp3",
            format="mp3",
        )


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)

st.markdown(
    """
# MeetingGPT

Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)


@st.cache_resource(show_spinner=False)
def generate_summary(transcript_path):
    loader = TextLoader(transcript_path)

    docs = loader.load_and_split(text_splitter=splitter)

    first_summary_prompt = ChatPromptTemplate.from_template(
        """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:
            """
    )

    first_summary_chain = first_summary_prompt | llm | StrOutputParser()

    summary = first_summary_chain.invoke(
        {"text": docs[0].page_content},
    )

    refine_prompt = ChatPromptTemplate.from_template(
        """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_summary}
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ------------
                {context}
                ------------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original summary.
                If there is no context to add, RETURN the existing summary.
                
                """
    )

    refine_chain = refine_prompt | llm | StrOutputParser()

    with st.status("Summarizing...") as status:
        for i, doc in enumerate(docs[1:]):
            status.update(
                label=f"Processing document {i+1}/{len(docs)-1} ")
            summary = refine_chain.invoke(
                {
                    "existing_summary": summary,
                    "context": doc.page_content,
                }
            )
            return summary

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
            """,
            ),
            ("human", "{question}"),
        ]
    )


with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

if video:
    chunks_folder = "./.cache/chunks"
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:
            f.write(video_content)
        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_folder, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())

    with summary_tab:
        if "summary" not in st.session_state:
            st.session_state.summary = ""

        start = st.button("Generate summary")
        if start:
            st.session_state.summary = generate_summary(transcript_path)
            st.write(st.session_state.summary)

    with qa_tab:
        retriever = embed_file(transcript_path)
        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        query = st.text_input("Ask a question about the video.")
        if query:
            send_message(query, "human")

            found = find_history(query)
            if found:
                send_message(found, "ai")
            else:
                chain = (
                    {
                        "docs": retriever,
                        "question": RunnablePassthrough(),
                    }
                    | RunnablePassthrough.assign(chat_history=load_memory)
                    | RunnableLambda(get_answers_no_src)
                    | RunnableLambda(choose_answer_no_src)
                )

                with st.chat_message("ai"):
                    result = chain.invoke(query)
                memory.save_context(
                    {"input": query},
                    {"output": result.content},
                )

else:
    st.session_state["messages"] = []
