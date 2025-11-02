import pickle
import sqlite3
from datetime import datetime
from typing import Sequence

from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage

SQLITE3_DB = './data.db'


def ensure_tables():
    with sqlite3.connect(SQLITE3_DB) as conn:
        conn.execute('pragma journal_mode = WAL;')
        conn.execute("""create table if not exists agent_conversation_history (
            thread_id     text     not null,
            mlflow_run_id text     not null,
            timestamp_utc datetime not null,
            chunk_id      integer  not null,
            messages      blob     not null
        )""")


def save_messages_to_db(
    *,
    thread_id: str,
    mlflow_run_id: str,
    timestamp_utc: datetime,
    chunk_id: int,
    messages: Sequence[AnyMessage],
):
    with sqlite3.connect(SQLITE3_DB) as conn:
        conn.execute('pragma journal_mode = WAL;')
        conn.execute(
            """
            insert into agent_conversation_history
            (thread_id, mlflow_run_id, timestamp_utc, chunk_id, messages)
            values
            (?, ?, ?, ?, ?);
            """,
            (thread_id, mlflow_run_id, timestamp_utc, chunk_id, pickle.dumps(messages)),
        )


def fetch_history_from_db(thread_id: str) -> Sequence[AnyMessage]:
    with sqlite3.connect(SQLITE3_DB) as conn:
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
    return compressed  # type: ignore
