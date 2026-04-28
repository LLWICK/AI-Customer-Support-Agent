from langgraph.graph import START,END, StateGraph
from states import agent_state
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.agents import create_agent
from retriever import retriever
from prompts import multi_query_prompt
load_dotenv()

llm = ChatGroq(model='openai/gpt-oss-120b')



def agent_Node(state: agent_state) -> agent_state:
    return {
        "messages": [llm.invoke(str(state['messages']))]
    }

def multi_query_agent_Node(state: agent_state) -> agent_state:
    return {
        "messages": [llm.invoke(str("provide me with 3 user queries simmilar to the given query: QUERY = "+str(state['messages'])))]
    }
 
builder = StateGraph(agent_state)

builder.add_node("agent_node", agent_Node)

builder.add_edge(START, "agent_node")
builder.add_edge("agent_node", END)

graph = builder.compile()

#Testing the various retrieval methods -JUST TESTING
query = "what are the careers available after completing the Bsc Hons in Data Science ?"



context = retriever(query)


prompt = f"Answer the given question ONLY using the given context QUESTION:{query}  CONTEXT: {context}"

result = graph.invoke({"messages": prompt})
print(result['messages'][-1].content)


