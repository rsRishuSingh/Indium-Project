'''
WorkFlow :

1. Read PDF
2. Create semantic chunks ( passing text page wise not whole document as once)
3. Storing Each chunk with metadata in qDB
4. Retrieving chunks based on similarity

'''
import os
import fitz
from qdrant_client import QdrantClient
from langchain.docstore.document import Document
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
# docker run -p 6333:6333 -v .:/qdrant/storage qdrant/qdrant

# Load embedding model from Hugging Face
embeddings_model  = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B"
)


INP_DIR = 'PDFs/'
BASE_EXT = '.pdf'
pdf_files = ["TESLA"]  

all_chunks = []
all_metadatas = []
all_docs = []


url="http://localhost:6333"
collection_name="q_DB_For_TESLA"

# threshold to make chunking balanced (ideal token/chunk size : 500 tokens for RAG)
def semantic_chunker(text, threshold=0.75): 
    chunker = SemanticChunker(embeddings_model, threshold=threshold)
    chunks = chunker.split_text(text)
    return chunks

def extract_chunks_from_pdf(pdf_name):
    pdf_path = INP_DIR + pdf_name + BASE_EXT
    doc = fitz.open(pdf_path)
    
    for page_index, page in enumerate(doc):
        
        page_no = page_index + 1
        text = page.get_text("text")
        chunks = SemanticChunker(text, 0.75)
        
        for chunk_index, chunk in enumerate(chunks):
             
             chunk_no = chunk_index + 1

             all_chunks.append(chunk)
             all_metadatas.append({
            "page": page_no + 1,
            "chunk_no": chunk_no,
            "source": f'{pdf_name}.pdf'
        })
    doc.close()


def create_documents():
     for pdf_name in pdf_files:
        extract_chunks_from_pdf(pdf_name)
        for i in range(len(all_chunks)):
            
            chunk = all_chunks[i]
            metadata = all_metadatas[i]
            doc =  Document( page_content=chunk,metadata=metadata)
            all_docs.append(doc)


def create_or_reload_qDB():
    client = QdrantClient(url=url, prefer_grpc=False)
    existing = [c.name for c in client.get_collections().collections]

    if collection_name not in existing:
        print(f"📦 Creating and seeding collection: {collection_name}")
        
        qdrant = Qdrant.from_documents(
        documents=all_docs,
        embedding=embeddings_model,
        url=url,
        prefer_grpc=False,
        collection_name=collection_name)

    else:
        print(f"✅ Using existing collection: {collection_name}")
    
        qdrant = Qdrant(
            client=client,
            embeddings=embeddings_model,
            collection_name=collection_name
        )

 
  






