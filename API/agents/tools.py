from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

search = GoogleSerperAPIWrapper()




def intermediate_answer(query: str) -> str:
    """Useful for when you need to ask with search."""
    return search.run(query)



