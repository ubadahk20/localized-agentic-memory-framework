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
