from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class agent_state(TypedDict):
    messages: Annotated[list, add_messages]