
def multi_query_prompt(query):
    return f'''provide me with 3 user queries similar to the given query only as an ARRAY  Of String.
                NO JARGON.
                           QUERY : {query}'''

def router_prompt(query: str):
    return f'''
    decide that if the task can be perform only using the basic llm agent
    or do you need to use the RAG knowledge base 

    Classify the query into one of:
    - rag (knowledge question)
    - agent (can answer to the query using just pure llm. No need of external knowledge)

    Query: {query}
    Answer only 'rag' or 'agent'.  NO JARGON

    '''

def rag_router_prompt(query: str, context: str ):
    return f'''
    Carefully examine the given QUESTION and CONTEXT very carefully
    decide if you can answer the question CORRECTLY using the provided context

    Decide if you want to use web search tool to answer the question correctly or
    the provided context is enough.

    classify the ability to answer the query with the provided context 
    and Classify this into one of :

    - web (the provided context in NOT sufficient enough to answer the question CORRECTLY and you need to use web search)
    - end (the provided context is sufficient enough to answer the question CORRECTLY)

    QUESTION : ${query}

    CONTEXT : ${context}

    Answer only 'web' or 'end'.  NO JARGON


'''