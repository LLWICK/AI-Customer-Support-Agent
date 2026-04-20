
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


loader = PyPDFLoader("API/test_docs/student_guide.pdf")
doc = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100 
)

chunks = splitter.split_documents(doc)

embedding = OllamaEmbeddings(model="nomic-embed-text")

vectorStore = FAISS.from_documents(chunks, embedding)
vectorStore.save_local('API/faiss_index')

def retriever(query: str):

    results = vectorStore.similarity_search(query=query, k=6)
    pairs = [(query, i.page_content) for i in results]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(results, scores))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

# Step 5: Pick top N
    top_docs = [doc for doc, score in scored_docs[:2]]

# Combine context
    context = "\n".join([doc.page_content for doc in top_docs])
    return context;


r= retriever("degree content of SLIIT data science degree")
print(r)



