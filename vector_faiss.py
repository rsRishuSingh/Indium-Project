from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

embedding_model = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

texts = [
    "The sun rises in the east.",
    "Python is a popular programming language.",
    "Delhi is the capital of India.",
    "OpenAI developed ChatGPT.",
     "Heart of India DElhi",
]


documents = [Document(page_content=text) for text in texts]

# By default, LangChain uses faiss.IndexFlatL2
# Flat: No compression or approximation. It stores all vectors in memory.
# L2: Distance metric is Euclidean distance.
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local("faiss_Vectorstore")

new_vectorstore = FAISS.load_local(
    "faiss_index_store",
      embedding_model,  
      allow_dangerous_deserialization=True)

query = "What is the capital of India?"
results = new_vectorstore.similarity_search(query, k=2)

for i, res in enumerate(results, 1):
    print(f"{i}. {res.page_content}")
