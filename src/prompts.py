
# prompts.py


def get_search_prompt(cleaned_input, formatted_result):
    """Generates the prompt template for standard web searches."""
    return (
        f"Instruction: Answer the user's query using only the search results provided below. "
        f"Be factual, direct, and concise. Do not guess or extrapolate if information is missing.\n\n"
        f"Query: {cleaned_input}\n\n"
        f"Search Results:\n---\n{formatted_result}\n---\n\n"
        f"Answer:"
    )


def get_deepsearch_prompt(cleaned_input, formatted_result):
    """Generates the prompt template for comprehensive web scrapes."""
    return (
        f"Instruction: Summarize the web scraping data below to answer the user's query. "
        f"Provide only the TOP 5 most relevant items as a brief bulleted list. "
        f"Do not repeat listings, and do not make up information.\n\n"
        f"User Query: {cleaned_input}\n\n"
        f"Data:\n====================\n{formatted_result}\n====================\n\n"
        f"Top 5 Bulleted Summary:"
    )


def get_incognito_prompt(cleaned_input):
    """Generates the prompt template for stateless safe sessions."""
    return (
        f"System Instruction: You are operating in a completely stateless, local environment. "
        f"Provide a highly direct, raw, and objective answer to the query below. "
        f"Omit all conversational pleasantries, introductory remarks, and structural disclaimers. "
        f"Focus entirely on the technical or factual details requested.\n\n"
        f"Query: {cleaned_input}\n\n"
        f"Answer:"
    )


def get_facts_prompt(transcript):
    """Prompt template for fact generation"""

    return (
        "You are a memory extraction system. You will be given a conversation "
        "transcript between a user and an AI assistant. Your only job is to extract "
        "durable, genuinely useful facts about the user — things worth remembering "
        "in future conversations.\n\n"
        "STRICT RULES:\n"
        "- IGNORE greetings, thank-yous, apologies, and small talk entirely.\n"
        "- IGNORE the assistant explaining its own capabilities or limitations.\n"
        "- IGNORE facts already obviously known or generic (e.g. 'the user said hi').\n"
        "- ONLY extract facts that are specific to this user: preferences, personal "
        "details, decisions, corrections, project details, or stated goals.\n"
        "- Do NOT invent or infer anything not explicitly stated in the transcript.\n"
        "- Each fact must be a single, standalone, atomic statement — one idea per fact.\n\n"
        "OUTPUT FORMAT:\n"
        "- Output ONLY a list of facts, one per line, each starting with '- '.\n"
        "- Do NOT include any preamble, explanation, or numbering.\n"
        "- If there are NO genuinely useful facts in this conversation, output exactly: NO_FACTS\n\n"
        "- Facts must be about the USER, never about the assistant itself."
        f"Conversation transcript:\n---\n{transcript}\n---\n\n"
        "Facts:"
    )


def get_remember_prompt(cleaned_input, retrieved_facts):
    return (
        f"Instruction: Answer the user's question using only the facts below, "
        f"which are things you remember about this user from past conversations. "
        f"Be direct and natural — don't mention 'the facts say' or reference this "
        f"as a lookup, just answer as if you recalled it.\n\n"
        f"Question: {cleaned_input}\n\n"
        f"Remembered facts:\n---\n{retrieved_facts}\n---\n\n"
        f"Answer:"
    )
