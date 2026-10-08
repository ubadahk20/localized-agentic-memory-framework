# Localized Agentic Memory Framework

Edge-deployed agentic memory system with dual-tier scheduled consolidation
and semantic deduplication, running entirely on local consumer hardware.

## Status: In Development (Final Year Project)

## Modules
- [x] Module 1: Local LLM setup (Ollama) + SQLite schema
- [x] Module 2: Core chat loop with ephemeral buffer
- [x] Module 3: Keyword routing (:search: / :remember:)
- [x] Module 4: Memory consolidation
- [x] Module 5: Three-layer semantic deduplication
- [x] Module 6: Testing & benchmarking

## Stack
Python, Ollama (Qwen-2.5-1.5B / Llama-3.2-3B), SQLite, ChromaDB

## Overview

The **Localized Agentic Memory Framework** is a local-first AI memory system designed to give an AI assistant persistent, structured memory while keeping all data and processing on the user's own machine.

The system uses a **dual-tier memory architecture**. Recent conversations are temporarily stored in an ephemeral SQLite buffer, while important information is extracted, consolidated, and stored as long-term semantic memories using ChromaDB.

The framework also includes **semantic deduplication**, which prevents the memory store from accumulating repeated or redundant information. Similar memories are compared using semantic similarity, while ambiguous cases are resolved using a local LLM.

All AI inference is performed locally through **Ollama**, using lightweight models suitable for consumer hardware. No cloud AI service or external account is required for the core memory system.

### Who Should Use It?

This project is primarily intended for:

* **Students and researchers** studying AI agents, memory systems, RAG, semantic search, or local LLMs.
* **Developers** who want to experiment with persistent memory for AI assistants.
* **Privacy-conscious users** who want an AI system that keeps conversations and memories on their own computer.
* **Edge-AI enthusiasts** interested in running agentic systems on consumer hardware without relying on cloud infrastructure.
* **Academic projects and demonstrations** involving LLM memory, semantic deduplication, SQLite, ChromaDB, and local inference.

It is particularly useful as a **research and educational framework** for understanding how an AI agent can move from short-term conversational context to persistent semantic memory.

---

## Basic Usage

### 1. Install the Requirements

Clone or download this repository and run the provided setup script.

The setup process installs the required Python packages and prepares the local environment.

> Make sure **Python** and **Ollama** are installed on your system before running the application.

### 2. Start the Required Local Models

The framework uses Ollama to run the local language models:

* **Qwen 2.5 1.5B** — used for fast conversational interaction.
* **Llama 3.2 3B** — used for memory extraction and more reasoning-intensive tasks.

The setup script can download the required models automatically.

### 3. Launch the Application

After setup, start the Streamlit application using the provided run script.

The application will open in your web browser.

### 4. Chat Normally

The **Chat** mode provides the standard conversational interface.

You can interact with the local AI just as you would with a normal chat assistant. Recent conversation turns are temporarily maintained in the session buffer.

### 5. Use Memory

When information should be retained as long-term memory, use the **Remember** functionality.

For example:

```text
Remember that my name is Ubadah.
```

The system extracts the relevant information and prepares it for long-term storage.

### 6. End the Session

When the session is finished, use:

**End Session & Save Memory**

This triggers the memory consolidation and deduplication pipeline.

The system:

1. Collects the relevant conversation information.
2. Extracts useful facts.
3. Compares new facts against existing memories.
4. Identifies duplicates and redundant information.
5. Resolves ambiguous cases using the local LLM.
6. Stores distinct or updated memories in the long-term semantic memory database.

### 7. Search and Recall

Previously stored information can be retrieved using the memory/search functionality.

Because memories are stored semantically, the system can retrieve information based on **meaning**, rather than requiring an exact keyword match.

For example, a stored memory such as:

```text
The user is studying Information Technology.
```

may be retrieved when asking:

```text
What does the user study?
```

even though the wording is different.

---

## Privacy & Local Processing

The framework is designed around a **local-first architecture**.

Conversation data, SQLite records, semantic memories, and LLM inference remain on the local machine.

The system does not require a cloud AI API or user account for its core functionality.

> **Your data stays on your machine.**

---

## Architecture at a Glance

```text
                User
                  │
                  ▼
          Streamlit Interface
                  │
                  ▼
             Chat / Router
                  │
          ┌───────┴────────┐
          │                │
          ▼                ▼
       SQLite           Ollama
   Session Buffer    Local LLM Models
          │
          ▼
   Memory Consolidation
          │
          ▼
   Semantic Deduplication
          │
          ▼
       ChromaDB
   Long-Term Memory
          │
          ▼
     Memory Retrieval
```

The overall pipeline can therefore be summarized as:

**Conversation → Temporary Buffer → Consolidation → Semantic Deduplication → Long-Term Memory → Retrieval**

---

## Project Completion

The project has successfully implemented the following modules:

* [x] Local LLM setup using Ollama
* [x] SQLite-based conversation buffer
* [x] Core conversational loop
* [x] Keyword-based routing
* [x] Memory consolidation
* [x] Semantic deduplication
* [x] Long-term semantic memory using ChromaDB
* [x] Testing and benchmarking
* [x] Streamlit user interface
* [x] Local-first execution

**Status: Completed — Final Year Project**
