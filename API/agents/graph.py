from langgraph.graph import START,END, StateGraph
from states import agent_state
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.agents import create_agent
from retriever import retriever
from prompts import *
from tools import *
load_dotenv()

llm = ChatGroq(model='openai/gpt-oss-120b')


def router_Node(state: agent_state)->agent_state:
    print("Router Node Executing...")
    result = llm.invoke(router_prompt(state['query']))
    print("Routing to : "+str(result.content))
    return{
        "route": result.content
    }


def RAG_agent_Node(state: agent_state) -> agent_state:
    print("RAG Agent Node Executing.... ")
    context = retriever(str(state['query']))
    
    print("RAG Agent context Retrieved.... ")
    return {
        "context" : context
        
    }

def Rag_router_Node(state: agent_state) ->agent_state:
    print("RAG Router Node Executing...")
    result = llm.invoke(rag_router_prompt(state['query'], state['context']))
    print("Routing to : "+str(result.content))
    return{
        "rag_route": result.content
    }

def web_search_agent_Node(state: agent_state) ->agent_state:
    print("Web Search Node Executing...")
    context = intermediate_answer(state['query'])
    
    return {
        "context": context
    }

    

    

def agent_Node(state: agent_state) -> agent_state:
    print("Executing agent Node.... ")
    return {
        "messages": [llm.invoke(f'''Answer the question according to your knowledge .
                                QUESTION : {state['query']} 
                                ''')]
    }


def route_decision(state: agent_state):
    return state["route"]

def rag_route_decision(state: agent_state):
    return state["rag_route"]

 
builder = StateGraph(agent_state)

builder.add_node("agent_node", agent_Node)
builder.add_node("RAG_node", RAG_agent_Node)
builder.add_node("Router_node", router_Node)
builder.add_node("Rag_Router_node", Rag_router_Node)
builder.add_node("Web_search_node", web_search_agent_Node)


builder.add_edge(START, "Router_node")

builder.add_conditional_edges(
    "Router_node",
    route_decision,
    {
        "rag": "RAG_node",
        "agent": "agent_node"
    }
)

builder.add_edge("RAG_node", "Rag_Router_node")

builder.add_conditional_edges(
    "Rag_Router_node",
    rag_route_decision,
    {
        "web": "Web_search_node",
        "end": "agent_node"
    }
)

builder.add_edge("Web_search_node", "agent_node")




builder.add_edge("agent_node", END)

graph = builder.compile()

#Testing the various retrieval methods -JUST TESTING
query = "who is the current Vice Chancellor Of SLIIT Malabe Campus in 2026?"





result = graph.invoke({"query": query})
print(result['messages'][-1].content)


