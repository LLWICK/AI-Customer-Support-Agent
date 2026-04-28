from langchain_groq import ChatGroq
from prompts import multi_query_prompt
from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(model='openai/gpt-oss-120b')

def multi_query_agent(query: str):
    prompt = multi_query_prompt(query)
    result = llm.invoke(prompt)
    return result.content;

#r = multi_query_agent("who are the foreign degree programmes organized by SLIIT ?")
#print(r)