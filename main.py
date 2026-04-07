#  using langchain-tavily feature for web search

from typing import Any, Dict, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
#from tavily import TavilyClient
from langchain_tavily import TavilySearch
from pydantic import Field



# Load environment variables from .env file
load_dotenv()

class Source(BaseModel):
    """Schema for a source used by a BaseModel"""
    url: str = Field(description="url of the source")

class AgentResponse(BaseModel):
    """schema for a agent to provide response and sources"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="list of sources used to generate answers")


#tavily = TavilyClient()

# @tool
# def search(query: str) -> Dict[str, Any]:
#     """Search the web for current information including weather, news, and other real-time data.
    
#     Use this tool when you need to find current information that you don't have access to,
#     such as weather conditions, current events, or any real-time data.
    
#     Args:
#        query: The search query string describing what information to find.
       
#     Returns:
#        The search results as a dictionary containing the requested information.
#     """
#     print(f"Searching for: {query}")
#     #return "Weather is cool today in New York"
#     return tavily.search(query=query)


tavily_tool = TavilySearch(max_results=5)
#tools = [search]
tools = [tavily_tool]
llm = ChatOllama(model="qwen2.5:3b", temperature=0)

agent = create_agent(model = llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain!")
    response=agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for  fresher for an ai engineer using langchain in India, Banglore/Bengaluru on linkedin and list their details")]})
    print(response)
   

if __name__ == "__main__":
    main()
