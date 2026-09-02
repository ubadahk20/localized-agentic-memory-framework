import ollama
import sqlite3
import uuid
import time
from datetime import datetime
from ddgs import DDGS
import trafilatura
import prompts

DB_PATH = "data/memory.db"
MODEL = "qwen2.5:1.5b"
session_id = str(uuid.uuid4())


def get_user_input():
    user_input = input("You: ").strip()
    return user_input


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn


conn = get_connection()


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
            return mode, cleaned_input

    return normal, user_input


def handle_search(cleaned_input):
    cleaned_result = []

    with DDGS() as ddgs:
        for r in ddgs.text(cleaned_input, max_results=10):
            title = r["title"]
            body = r["body"]
            cleaned_result.append(f"{title}: {body}")
    formatted_result = "\n".join(cleaned_result)
    prompt = prompts.get_search_prompt(cleaned_input, formatted_result)
    response = ollama.chat(model=MODEL, messages=[{"role": "user",
                                                   "content": prompt}])

    reply = response["message"]["content"]
    return reply


def handle_deepsearch(cleaned_input):
    cleaned_result = []
    with DDGS() as ddgs:
        for r in ddgs.text(cleaned_input, max_results=10):
            url = r["href"]
            downloaded = trafilatura.fetch_url(url)
            result = trafilatura.extract(downloaded)
            if result is not None:
                shortened_text = result[:1000]
                cleaned_result.append(f"{shortened_text}")
    formatted_result = "\n".join(cleaned_result)
    prompt = prompts.get_deepsearch_prompt(cleaned_input, formatted_result)
    response = ollama.chat(model=MODEL, messages=[
                           {"role": "user", "content": prompt}])

    reply = response["message"]["content"]
    return reply


def handle_remember(cleaned_input):
    return f"[No long-term memory system yet — this will query ChromaDB once Module 5 is built. Your query was: '{cleaned_input}']"


def handle_incognito(cleaned_input):
    cleaned_result = []

    with DDGS() as ddgs:
        for r in ddgs.text(cleaned_input, max_results=10):
            title = r["title"]
            body = r["body"]
            cleaned_result.append(f"{title}: {body}")
    formatted_result = "\n".join(cleaned_result)
    prompt = prompts.get_incognito_prompt(cleaned_input)
    response = ollama.chat(model=MODEL, messages=[{"role": "user",
                                                   "content": prompt}])

    reply = response["message"]["content"]
    return reply


def handle_chat(cleaned_input):

    history = load_history(conn, session_id)
    print("--- DEBUG: sending this history ---")
    print(history)

    response = ollama.chat(model=MODEL, messages=history)
    print("--- DEBUG: raw response ---")
    print(response)

    reply = response["message"]["content"]
    return reply


'''handlers is a dict that calls respective handler func
    based on mode returned by route_message()'''


'''main() is the entry point for the chat application '''


def main():

    print(f'Session started: {session_id}')
    print('Type "exit" to end the session.\n')
    handlers = {"search": handle_search, "deepsearch": handle_deepsearch,
                "remember": handle_remember, "incognito": handle_incognito, "chat": handle_chat}

    print("Warming up model...")
    ollama.chat(model=MODEL, messages=[{"role": "user", "content": "hi"}])
    print("Ready.\n")

    while True:
        user_input = get_user_input()
        if user_input.lower() == "exit":
            break

        infer_start = time.perf_counter()

        mode, cleaned_input = route_message(user_input)

        if mode != "incognito":
            user_write_ms = save_turn(conn, session_id, "user", user_input)

        handler_func = handlers.get(mode, handle_chat)
        try:
            reply = handler_func(cleaned_input)
        except Exception as e:
            print(f"Error occurred while handling {mode}: {e}")
            reply = "Sorry, I encountered an error while processing your request."
        print(f"Assistant: {reply}")
        infer_ms = (time.perf_counter() - infer_start) * 1000

        if mode != "incognito" and reply.strip() != '':
            assistant_write_ms = save_turn(
                conn, session_id, "assistant", reply)
            write_ms = assistant_write_ms + user_write_ms
        else:
            write_ms = 0

        print(
            f"  [buffer write: {write_ms:.2f}ms | model inference: {infer_ms:.1f}ms]\n")


if __name__ == "__main__":
    main()
