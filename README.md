# AI Research Agent with Tool Calling

This project is a hands-on implementation of an AI research agent built using **LangChain**, **Claude (Anthropic)**, and external tools like **DuckDuckGo Search** and **Wikipedia**.

The agent:
- Understands a research query
- Uses tools to gather information
- Produces **structured JSON output** using Pydantic
- Tracks which tools were used
- Can persist results to a local file

## Features
- Tool-calling AI agent
- Web search integration
- Wikipedia lookup
- Structured outputs (Pydantic)
- Clean separation of tools and agent logic
- Safe handling of API keys via `.env`

## Tech Stack
- Python
- LangChain (v1.2.3)
- Anthropic Claude
- Pydantic
- DuckDuckGo & Wikipedia tools

## How to Run
```bash
pip install -r requirements.txt
python main.py
