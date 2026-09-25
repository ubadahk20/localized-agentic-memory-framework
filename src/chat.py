import ollama
import sqlite3
import uuid
import time
from datetime import datetime
import consolidate
import router as rtr
import dedup

DB_PATH = "data/memory.db"
# MODEL_FAST = "qwen2.5:1.5b"
MODEL_EXTRACTION = "llama3.2:3b"
session_id = str(uuid.uuid4())


def get_user_input():
    user_input = input("You: ").strip()
    return user_input


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn


conn = get_connection()


def user_feedback():
    feedback = input(
        "Save this interaction to memory? yes/1 [default], no/0: ").lower().strip()
    flag = 0 if feedback in ["0", "no", "n"] else 1
    return flag

# The save_turn() function is a database logger


def save_turn(conn, session_id, role, content):
    start = time.perf_counter()
    conn.execute(
        "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()

    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms

# handle_chat() is the default handler for normal chat mode


def handle_chat(cleaned_input):

    history = rtr.load_history(conn, session_id)
    # print("--- DEBUG: sending this history ---")
    # print(history)

    response = ollama.chat(model=MODEL_EXTRACTION, messages=history)
    # print("--- DEBUG: raw response ---")
    # print(response)

    reply = response["message"]["content"]
    return reply


# main() is the entry point for the chat application

def main():

    print(f'Session started: {session_id}')
    print('Type "exit" to end the session.\n')

    # handlers is a dict that calls respective handler func
    # based on mode returned by route_message()

    handlers = {"search": rtr.handle_search, "deepsearch": rtr.handle_deepsearch,
                "remember": rtr.handle_remember, "incognito": rtr.handle_incognito, "chat": handle_chat}

    print("Warming up model...")
    ollama.chat(model=MODEL_EXTRACTION, messages=[
                {"role": "user", "content": "hi"}])
    print("Ready.\n")

    while True:
        user_input = get_user_input()
        if user_input.lower() == "exit":
            consolidate.trigger_consolidation(conn, session_id)
            dedup.trigger_deduplication(conn)
            break

        infer_start = time.perf_counter()
        mode, cleaned_input = rtr.route_message(user_input)

        handler_func = handlers.get(mode, handle_chat)
        try:
            reply = handler_func(cleaned_input)
        except Exception as e:
            print(f"Error occurred while handling {mode}: {e}")
            reply = "Sorry, I encountered an error while processing your request."

        infer_ms = (time.perf_counter() - infer_start) * 1000

        # 1. Display assistant response first
        print(f"Assistant: {reply}")

        # 2. Ask user for feedback immediately after response
        flag = user_feedback() if mode != "incognito" else 0

        # 3. Save BOTH user and assistant turns if feedback is positive
        if mode != "incognito" and reply.strip() != '' and flag != 0:
            user_write_ms = save_turn(conn, session_id, "user", user_input)
            assistant_write_ms = save_turn(
                conn, session_id, "assistant", reply)
            write_ms = user_write_ms + assistant_write_ms
        else:
            write_ms = 0

        print(
            f"  [buffer write: {write_ms:.2f}ms | model inference: {infer_ms:.1f}ms]\n")

    conn.close()


if __name__ == "__main__":
    main()
