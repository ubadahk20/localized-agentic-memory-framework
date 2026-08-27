import ollama
import sqlite3
import uuid
import time
from datetime import datetime

DB_PATH = "data/memory.db"
MODEL = "qwen2.5:1.5b"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn


'''The save_turn() function is a database logger'''


def save_turn(conn, session_id, role, content):
    start = time.perf_counter()

    conn.execute(
        "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()

    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms


'''load_history() is a database retriever'''


def load_history(conn, session_id):
    rows = conn.execute(
        "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY id",
        (session_id,)
    ).fetchall()
    return [{'role': r,  'content': c} for r, c in rows]


modes = [(":search:", "search"), (":deepsearch:", "deepsearch"),
         (":remember:", "remember"), (":incognito:", "incognito")]
normal = "chat"
'''route_message() tells us which mode to use based on the user input prefix'''


def route_message(user_input):
    for prefix, mode in modes:
        if user_input.startswith(prefix):
            cleaned_input = user_input.removeprefix(prefix).strip()
            print(mode, cleaned_input)
            return mode, cleaned_input

    print(normal, user_input)
    return normal, user_input


def handle_search(cleaned_input):
    print("search handler called")


def handle_deepsearch(cleaned_input):
    print("deepsearch handler called")


def handle_remember(cleaned_input):
    print("remember handler called")


def handle_incognito(cleaned_input):
    print("incognito handler called")


def handle_chat(cleaned_input):
    print("chat handler called")


'''handlers is a dict that calls respective handler func based on mode returned by route_message()'''

handlers = {"search": handle_search, "deepsearch": handle_deepsearch,
            "remember": handle_remember, "incognito": handle_incognito, "chat": handle_chat}

mode, cleaned_input = route_message(":remember: The Amazing Spiderman")
handler_func = handlers.get(mode, handle_chat)
handler_func(cleaned_input)


'''main() is the entry point for the chat application '''


def main():
    session_id = str(uuid.uuid4())
    conn = get_connection()
    print(f'Session started: {session_id}')
    print('Type "exit" to end the session.\n')

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        write_ms = save_turn(conn, session_id, "user", user_input)

        history = load_history(conn, session_id)

        infer_start = time.perf_counter()
        response = ollama.chat(model=MODEL, messages=history)
        infer_ms = (time.perf_counter() - infer_start) * 1000

        reply = response["message"]["content"]
        print(f'Assistant: {reply}')

        save_turn(conn, session_id, "Assistant", reply)

        print(
            f"  [buffer write: {write_ms:.2f}ms | model inference: {infer_ms:.1f}ms]\n")

    conn.close()


if __name__ == "__main__":
    main()
