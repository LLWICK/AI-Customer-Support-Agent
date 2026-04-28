from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

class agent_state(TypedDict):
    messages: Annotated[list, add_messages];