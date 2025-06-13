# docker run -p 6333:6333 -v .:/qdrant/storage qdrant/qdrant
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

emb = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

url="http://localhost:6333"
collection_name="q_DB"

qdrant = Qdrant.from_texts(
    texts=["my name is Rishu",],
    metadatas = [{"source": "user_profile", "type": "bio"}],
    embedding=emb,
    url=url,
    prefer_grpc=False,
    collection_name=collection_name,
)
# print(qdrant)

# docs = [
#     Document(
#         page_content="My name is Rishu",
#         metadata={"source": "user_profile", "type": "bio"}
#     )
# ]

# qdrant = Qdrant.from_documents(
#     documents=docs,
#     embedding=embedding,
#     url="http://localhost:6333",
#     prefer_grpc=False,
#     collection_name="q_DB"
# )

# results = qdrant.similarity_search("Who is Rishu?", k=3)

# for i, doc in enumerate(results):
#     print(f"--- Result {i+1} ---")
#     print("Text:", doc.page_content)
#     print("Metadata:", doc.metadata)


# official qDB client ( not from langchain wrapper )


# retrieve using qdrant client offiial client of Qdrant
client = QdrantClient(
    url="http://localhost:6333",
    prefer_grpc=False
)
collections = client.get_collections()
# print(collections)

db = Qdrant(
    client = client,
    embeddings= emb,
    collection_name= collection_name
)

# print(db)

query = 'what is my name'

results = db.similarity_search(
    query = query,
    k = 3
)
for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print("Text:", doc.page_content)
 

