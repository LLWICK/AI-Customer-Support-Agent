
def multi_query_prompt(query):
    return f'''provide me with 3 user queries similar to the given query only as an ARRAY  Of String.
                NO JARGON.
                           QUERY : {query}'''