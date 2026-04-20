from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from states import *
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

def agent_node(state: agent_state)-> agent_state:
    print("agent Node executing....")
    return {
        "messages": [llm.invoke(state["messages"])]
    }

def tool_agent(state: agent_state)->agent_state:
    print("Tool Agent executing....")
    return {
        "messages": ["Tool Agent Response"]
    }

def RAG_agent(state: agent_state)->agent_state:
    print("Tool Agent executing....")
    return {
        "messages": ["Tool Agent Response"]
    }


builder = StateGraph(agent_state)

builder.add_node("agent_node", agent_node)
builder.add_node("RAG_agent", RAG_agent)

builder.add_edge(START, "agent_node")
builder.add_edge("agent_node", "RAG_agent")
builder.add_edge("RAG_agent",END)

graph = builder.compile()

result = graph.invoke({"messages":"Who is the president of US?"})
print(result["messages"][-1].content)
