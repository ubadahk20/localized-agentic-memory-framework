
import sqlite3
import ollama
import prompts
import uuid
import hashlib
MODEL = "qwen2.5:1.5b"

DB_PATH = "data/memory.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn


def get_unconsolidated_session_ids(conn):
    source_session_id = conn.execute(
        "SELECT DISTINCT session_id FROM conversations WHERE consolidated = 0 ").fetchone()
    # print(source_session_id)
    # Extract session_id
    return source_session_id[0] if isinstance(source_session_id, tuple) else None


def get_ids(conn, source_session_id):
    ids = conn.execute(
        "SELECT id FROM conversations WHERE session_id = ? AND consolidated = 0 ORDER BY id LIMIT 50", (
            source_session_id,)
    ).fetchall()
    # print(ids)
    return [row[0] for row in ids]  # Extract ids


def get_session_details(conn, source_session_id):
    rows = conn.execute(
        "SELECT id, role, content FROM conversations WHERE session_id = ? AND consolidated = 0 ORDER BY id LIMIT 50", (
            source_session_id,)
    ).fetchall()
    return rows


def format_rows_as_transcript(rows):
    lines = []
    for row_id, role, content in rows:
        lines.append(f"{role.upper()}: {content}")
    # print("\n".join(lines))
    return "\n".join(lines)


def get_facts(transcript):
    prompt = prompts.get_facts_prompt(transcript)
    facts = ollama.chat(model=MODEL, messages=[
                        {"role": "system", "content": prompt}])
    reply = facts['message']['content']

    if reply.strip() == "NO_FACTS":
        return None

    lines = reply.split("\n")
    fact_list = [line.lstrip("- ").strip()
                 for line in lines if line.strip() != ""]
    return fact_list


def hash_converter(fact_list):
    fact_data = []
    for line in fact_list:
        hash_object = hashlib.sha256(line.encode())
        hash_value = hash_object.hexdigest()
        # print(f"Hash value for '{line}': {hash_value}")
        fact_data.append((line, hash_value))
        # print(fact_data)
    return fact_data


def save_facts(conn, source_session_id, fact_data):
    for reply, hash_value in fact_data:
        existing = conn.execute(
            "SELECT fact_hash fROM facts WHERE fact_hash = ?", (hash_value,)
        ).fetchone()

        if existing:
            print(f"Skipping exact duplicate: {reply}")
            continue
        else:
            conn.execute(
                "INSERT INTO facts (fact_text, source_session_id, fact_hash) VALUES (?, ?, ?)", (
                    reply, source_session_id, hash_value)
            )
    conn.commit()
    print("success")


def update_consolidated_status(conn, source_session_id, ids):
    for id in ids:
        conn.execute(
            "UPDATE conversations SET consolidated = 1 WHERE session_id = ? AND id = ?", (
                source_session_id, id)
        )
    print(
        f"Updated consolidated status for session_id: {source_session_id} and ids: {ids}")
    conn.commit()


def check_saved_facts(conn, source_session_id):
    rows = conn.execute(
        "SELECT id FROM facts WHERE source_session_id = ?", (
            source_session_id,)
    ).fetchone()
    if rows:
        return True
    else:
        print(f"No facts found with ID: {source_session_id}")
        return False


def del_raw_conversations(conn, source_session_id):
    conn.execute(
        "DELETE FROM conversations WHERE session_id = ? AND consolidated = 1", (
            source_session_id,)
    )
    conn.commit()


def trigger_consolidation(conn, session_id):
    # unconsolidated = get_unconsolidated_session_ids(conn)
    ids = get_ids(conn,
                  session_id)
    session_rows = get_session_details(conn, session_id)
    transcript = format_rows_as_transcript(session_rows)
    reply = get_facts(transcript)
    if reply is None:
        print("No facts to save for this session.")
        update_consolidated_status(
            conn, session_id, ids)  # still mark it done
    else:
        fact_data = hash_converter(reply)
        save_facts(conn, session_id, fact_data)
        update_consolidated_status(
            conn, session_id, ids)
        if check_saved_facts(conn, session_id):
            del_raw_conversations(conn, session_id)

    conn.close()


if __name__ == "__main__":
    trigger_consolidation()
