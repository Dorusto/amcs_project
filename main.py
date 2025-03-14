from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool, pdf_query_tool
import json

# Load the environment variables
load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Create a class to hold the request data
llm = ChatOpenAI(model="gpt-4o-mini")
# llm = ChatAnthropic(model='claude-3-5-sonnet-20241022')

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Analyze the AMCS system documentation and accurately respond to user questions based on its content. When answering, follow these guidelines:
            Extract and present only the relevant information from the documentation that directly answers the user’s query.
            Maintain technical accuracy by using the exact terminology and descriptions found in the manual.
            If the question is about a specific system component (e.g., pumps, valves, alarms, etc.), provide a detailed explanation of its operation, control methods, alarms, and settings.
            If applicable, include step-by-step instructions for interacting with the system via the graphical user interface (GUI).
            If the query involves troubleshooting, describe potential issues, alarms, and solutions provided in the manual.
            Avoid summarizing too much—deliver a response that reflects the full depth of the documentation where necessary.
            whoIf the requested information is not found in the documentation, clearly state that it is not available instead of generating an assumption.\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool, pdf_query_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you research? ")
raw_response = agent_executor.invoke({"query": query})

import json

try:
    # Extragem output-ul
    output_str = raw_response.get("output", "")

    # Eliminăm delimitatorii specifici LangChain care încadrează JSON-ul
    cleaned_output = output_str.strip("```json\n").strip("\n```")

    # Parsăm JSON-ul
    parsed_response = json.loads(cleaned_output)

    print(parsed_response)  # Obiect JSON corect
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    print(f"Raw Response: {raw_response}")
