from .chat_handler import memory
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")

    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
    )


def find_history(query):
    histories = memory.load_memory_variables({})["chat_history"]
    temp = []
    for idx in range(len(histories) // 2):
        temp.append(
            {
                "input": histories[idx * 2].content,
                "output": histories[idx * 2 + 1].content,
            }
        )

    docs = [
        Document(
            page_content=f"input:{item['input']}\noutput:{item['output']}")
        for item in temp
    ]
    try:
        # Create vector store from the documents
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())

        # Perform similarity search
        found_docs_with_scores = vector_store.similarity_search_with_relevance_scores(
            query, k=3)

        # Define a threshold for the score
        threshold = 0.8
        relevant_docs = [doc for doc,
                         score in found_docs_with_scores
                         if score >= threshold]
        # If a relevant document is found, return the most similar one
        if relevant_docs:
            candidate = relevant_docs[0].page_content.split("\n")[1]
            return candidate.replace("output:", "")
        else:
            return None
    except IndexError:
        return None


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 1
    docs = loader.load_and_split(splitter)
    vector_store = FAISS.from_documents(
        docs,
        OpenAIEmbeddings(),
    )

    return vector_store.as_retriever()
