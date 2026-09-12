"""
Backend simples de conversa com memória persistente em SQLite.
Usado como base para o grafo principal do agente de blog.
"""
import os
import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

DB_PATH = os.getenv("CHATBOT_DB_PATH", "chatbot.db")


class ChatState(TypedDict):
    """Estado do grafo simples de conversa."""
    messages: Annotated[list[BaseMessage], add_messages]


def build_simple_graph(model_name: str = "gpt-4o-mini"):
    """Constrói um StateGraph simples com memória SQLite."""
    llm = ChatOpenAI(model=model_name, temperature=0.3, streaming=True)

    def chatbot_node(state: ChatState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return builder.compile(checkpointer=checkpointer)