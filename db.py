import pickle
import sqlite3
from datetime import datetime
from typing import Sequence

from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage

SQLITE3_DB = './data.db'


def save_messages_to_db(
    *,
    thread_id: str,
    mlflow_run_id: str,
    subject: str,
    timestamp_utc: datetime,
    chunk_id: int,
    model_repr: str,
    messages: Sequence[AnyMessage],
):
    with sqlite3.connect(SQLITE3_DB) as conn:
        conn.execute('pragma journal_mode = WAL;')
        conn.execute(
            """
            insert into agent_conversation_history
            (thread_id, mlflow_run_id, subject, timestamp_utc, chunk_id, model_repr, messages)
            values
            (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                thread_id,
                mlflow_run_id,
                subject,
                timestamp_utc,
                chunk_id,
                model_repr,
                pickle.dumps(messages),
            ),
        )
        conn.commit()


def fetch_history_from_db(thread_id: str) -> Sequence[AnyMessage]:
    with sqlite3.connect(SQLITE3_DB) as conn:
        conn.execute('pragma journal_mode = WAL;')
        records = conn.execute(
            """
            select timestamp_utc, messages
            from agent_conversation_history
            where thread_id = ?
            order by timestamp_utc asc
            """,
            (thread_id,),
        ).fetchall()
    messages: Sequence[HumanMessage | AIMessageChunk] = [
        msg
        for _, blob in sorted(records, key=lambda record: record[0])
        for msg in pickle.loads(blob)
    ]
    compressed: Sequence[HumanMessage | AIMessage] = []
    current_running_ai_messages: Sequence[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            if current_running_ai_messages:
                compressed.append(AIMessage(content=''.join(current_running_ai_messages)))
                current_running_ai_messages.clear()
            compressed.append(message)
        elif isinstance(message, AIMessage):
            current_running_ai_messages.append(str(message.content))
    if current_running_ai_messages:
        compressed.append(AIMessage(content=''.join(current_running_ai_messages)))
        current_running_ai_messages.clear()
    return compressed
