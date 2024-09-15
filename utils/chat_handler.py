from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory

from utils.utils import ChatCallbackHandler


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4",
)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=150,
    memory_key="chat_history",
    return_messages=True,
    human_prefix="User",
    ai_prefix="AI",
)

answers_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Using ONLY the following context, answer the user's question. Do not include previous AI responses in your answer. If you don't know the answer, just say you don't know; don't make anything up.

            Then, give a score to the answer between 0 and 5.
            If the answer addresses the user's question, the score should be high; otherwise, it should be low.
            Make sure to always include the answer's score, even if it's 0.
            Context: {context}

            Examples:

            Question: How far away is the moon?
            Answer: The moon is 384,400 km away.
            Score: 5

            Question: How far away is the sun?
            Answer: I don't know
            Score: 0    
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def get_answers(input):
    retriever = input["docs"]
    question = input["question"]
    chat_history = input["chat_history"]

    # Use the retriever to get relevant documents for the question
    docs = retriever.get_relevant_documents(question)

    user_messages = [msg for msg in chat_history if msg.type ==
                     "human" or msg.role == "user"]

    llm_local = ChatOpenAI(
        temperature=0.1,
        model="gpt-4",
        streaming=False,
        callbacks=None
    )

    answers_chain = answers_prompt | llm_local

    return {
        "question": question,
        "chat_history": user_messages,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                        "chat_history": user_messages,
                    }
                ).content,
                "source": doc.metadata.get("source", "Unknown"),
            }
            for doc in docs
        ],
    }


def get_answers(input):
    docs = input["docs"]
    question = input["question"]
    chat_history = input["chat_history"]

    user_messages = [
        msg for msg in chat_history
        if msg.type == "human" or getattr(msg, "role", None) == "user"
    ]

    llm_local = ChatOpenAI(
        temperature=0.1,
        model="gpt-4",
        streaming=False,
        callbacks=None
    )
    answers_chain = answers_prompt | llm_local
    return {
        "question": question,
        "chat_history": user_messages,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                        "chat_history": user_messages,
                    }
                ).content,
                "source": doc.metadata["source"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Do not include previous AI responses in your answer.
            Cite sources and return the sources of the answers as they are; do not change them.
            Answers: {answers}
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

choose_prompt_no_src = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Do not include previous AI responses in your answer.
            
            Answers: {answers}
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    chat_history = inputs["chat_history"]

    user_messages = [msg for msg in chat_history if msg.type ==
                     "human" or msg.role == "user"]

    llm_local = ChatOpenAI(
        temperature=0.1,
        model="gpt-4",
        streaming=True,
        callbacks=[ChatCallbackHandler()]
    )

    choose_chain = choose_prompt | llm_local

    condensed = "\n\n".join(
        f"Answer: {answer['answer']}\nSource: {answer['source']}\n"
        for answer in answers
    )

    return choose_chain.invoke(
        {
            "answers": condensed,
            "question": question,
            "chat_history": user_messages,
        }
    )


def choose_answer_no_src(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    chat_history = inputs["chat_history"]

    llm.streaming = True
    llm.callbacks = [ChatCallbackHandler()]

    choose_chain = choose_prompt_no_src | llm

    condensed = "\n\n".join(
        f"Answer: {answer['answer']}\n"
        for answer in answers
    )

    return choose_chain.invoke(
        {
            "answers": condensed,
            "question": question,
            "chat_history": chat_history,
        }
    )
