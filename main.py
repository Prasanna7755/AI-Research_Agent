import os
import json
from typing import Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

#from langchain.agents.executor import AgentExecutor
#from langchain.agents.tool_calling_agent import create_tool_calling_agent

from tools import TOOLS

# 1. Load environment variables from .env file
load_dotenv()  # Load environment variables from .env file

#2. Define structured output schema

class ResearchResponse(BaseModel):
    topic: str = Field(description="Short title / topic of the research paper")
    summary: str = Field(description="A concise summary of the research findings / answer")
    sources: list[str] = Field(description="List of sourcesused (URLs or source names)")
    tools_used: list[str] = Field(description="List of tools used in the research")

#3. Agent Step Decision Schema
class AgentDecision(BaseModel):
    action: Literal["tool", "final"] = Field(description="Whether to call a tool or produce the final answer")
    tool_name: Optional[str] = Field(default=None, description="Tool name if action=tool")
    tool_input: Optional[str] = Field(default=None, description="Input to the tool if action=tool")

def main():

    #4. Initialize the language model and prompt template
    #llm = ChatOpenAI(model_name="gpt-4o-mini")
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)  # Example model name
    #response = llm.invoke("When did India get Independence?")
    #print(response.content)

    #5. Tool Registry
    tool_map = {t.name: t for t in TOOLS}
    tool_descriptions = "\n".join([f"{t.name}: {t.description}" for t in TOOLS])    

    #6. Parser to force valid structured output
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)

    #7. Prompt with schema instructions
    final_parsers = PydanticOutputParser(pydantic_object=ResearchResponse)
    decision_parser = PydanticOutputParser(pydantic_object=AgentDecision)


    #Prompt for deciding tool vs final
    decision_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an AI research agent. \n"
                    "You can use tools to gather information.\n\n"
                    "Available tools:\n"
                    f"{tool_descriptions}\n\n"
                    "Decide the next step.\n"
                    "If you need external info, choose action=tool and specify tool_name and tool_input.\n"
                    "If you have enough info to answer, choose action=final.\n\n"
                    "Return ONLY valid JSON that matches the schema:\n"
                    "{format_instructions}"
                ),
            ),
            ("human", "User Question: \n{query}\n\nCurrent notes (maybe empty): \n{notes}\n"),        ]
    ).partial(format_instructions=parser.get_format_instructions())

    #Prompt for final structured answer
    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a research assistant. \n"
                    "Use the notes to write the final answer. \n"
                    "Return ONLY valid JSON matching this schema:\n"
                    "{format_instructions}\n"
                    "No extra text."
                ),
            ),
            ("human", "Here are your research notes:\n{notes}\n"),
        ]
    ).partial(format_instructions=final_parsers.get_format_instructions())

    query = ("Provide a detailed research summary on the impact of climate change on marine biodiversity.")

    notes=""
    tools_used = []

    #Simple agent loop
    for _ in range(3):
        try:
            decision: AgentDecision = (decision_prompt | llm | decision_parser).invoke(
                {"query": query, "notes": notes}
            )
        except Exception as e:
            print(f"\n Failed to Parse agent decision: {e}")
            raw = (decision_prompt | llm).invoke({"query": query, "notes": notes})
            print("\n Raw decision output: \n", raw.content if hasattr(raw, "content") else raw)
            return

        if decision.action == "final":
            break

        if decision.action == "tool":
            if not decision.tool_name or not decision.tool_input:
                notes += "\n[Agent Issue] Tool step missing tool_name or tool_input\n"
                break
        
            if decision.tool_name not in tool_map:
                notes += f"\n [Agent Issue] Unknown tool: {decision.tool_name}\n"
                break

        tool = tool_map[decision.tool_name]
        try:
            result = tool.func(decision.tool_input)
        except Exception as e:
            result = f"[Tool Error] {e}"

        tools_used.append(decision.tool_name)
        notes += f"\n[Tool: {decision.tool_name} | Input: {decision.tool_input}]\nResult: {result}\n"

    #Produce final structured output
    try: 
        final: ResearchResponse = (final_prompt | llm | final_parsers).invoke({"query": query, "notes": notes})
        #Ensure tools_used reflects reality
        final.tools_used = sorted(list(set(tools_used)))
        print(json.dumps(final.model_dump(), indent=2))
    except Exception as e:
        print("\n Failed to Parse final response: {e}")
        raw = (final_prompt | llm).invoke({"query": query, "notes": notes})
        print("\n Raw final output: \n", raw.content if hasattr(raw, "content") else raw)

if __name__ == "__main__":
    main()

