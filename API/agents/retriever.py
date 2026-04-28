from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from agents import multi_query_agent

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')




embedding = OllamaEmbeddings(model="nomic-embed-text")



loader = PyPDFLoader(file_path="C:/Users/CHAMA COMPUTERS/Desktop/Data_Science/AI_ML/projects/AI_customer_support_agent/AI-Customer-Support-Agent/API/test_docs/student_guide.pdf")

splitter = RecursiveCharacterTextSplitter(chunk_size = 700, chunk_overlap = 50 )



docs = loader.load()

chunked_doc= splitter.split_documents(docs)



tokenized_Corpus = [i.page_content.lower().split()  for i in chunked_doc]

bm25 = BM25Okapi(tokenized_Corpus)

def bm25_search(query, k=5):
    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    # Get top k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [chunked_doc[i] for i in top_indices]

vectorStore = FAISS.from_documents(chunked_doc, embedding)

vectorStore.save_local("/API/FAISS_INDEX")



def retriever(query: str):

    queries = multi_query_agent(query)
    

    sementic_retrieved_docs = vectorStore.similarity_search(query=query, k=7);

    bm25_results = bm25_search(query, k=5)

    all_results = bm25_results + sementic_retrieved_docs

    unique_docs = list({
        doc.page_content: doc for doc in all_results
    }.values())

    pairs = [(query,i.page_content) for i in unique_docs]
    ranking = reranker.predict(pairs)

    scored_docs = list(zip(unique_docs, ranking))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

# Step 5: Pick top N
    top_docs = [doc for doc, score in scored_docs[:5]]
    
    jn = "/n".join([x.page_content for x in top_docs])
    return jn;

#results = retriever("SLIIT codefest ")
#print(results)



