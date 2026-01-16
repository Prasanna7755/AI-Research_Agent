from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from datetime import datetime

#---- Web Search Tool ----#

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name ="search",
    func=search.run,
    description="Search the web for relevent and recent information on a given topic.",
)

#---- Wikipedia Tool ----#

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

wiki_tool = Tool(
    name="wikipedia",
    func=wiki.run,
    description="Look up a topic on Wikipedia and return a concise summary",
)

#---- Save to TXT Tool - Custom Tool ----#

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n"
    with open(filename, "a", encoding="utf-8") as file:
        file.write(formatted_text)
        file.write("\n\n")  # Add extra newlines for separation

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves the provided research output text to a local .txt file with a timestamp.",
)
TOOLS = [search_tool, wiki_tool, save_tool]