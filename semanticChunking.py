# Install required packages
# !pip install -qU langchain langchain-experimental transformers accelerate sentence-transformers

import torch
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List

# Custom wrapper for Qwen embeddings with instruction prefixes
class QwenEmbeddingsWrapper(HuggingFaceEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Add document prefix to input texts"""
        prefix = "Represent the document for retrieval: "
        prefixed_texts = [prefix + text for text in texts]
        return super().embed_documents(prefixed_texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Add query prefix to input text"""
        prefix = "Instruct: "
        return super().embed_query(prefix + text)

# Initialize embeddings
model_name = "Qwen/Qwen3-Embedding-0.6B"
embeddings = QwenEmbeddingsWrapper(
    model_name=model_name,
    model_kwargs={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'trust_remote_code': True
    },
    encode_kwargs={
        'batch_size': 4,
        'show_progress_bar': True,
        'normalize_embeddings': True  # Critical for semantic chunking
    }
)

# Verify embedding functionality
test_text = "LangChain makes LLM applications easier to build"
embedding = embeddings.embed_query(test_text)
print(f"Embedding dimension: {len(embedding)}")  # Should be 2048

# Semantic chunking function
def semantic_split(text, embeddings, **kwargs):
    """Split text using semantic boundaries"""
    chunker = SemanticChunker(embeddings=embeddings, **kwargs)
    return chunker.split_text(text)

# Example usage
if __name__ == "__main__":
    sample_text = """
    Large language models (LLMs) are artificial intelligence tools 
    that can read, summarize and translate texts and predict future 
    words in a sentence letting them generate sentences similar to 
    how humans talk and write. This transformative technology has 
    applications across numerous domains including content creation, 
    customer support automation, and data analysis. When working with 
    LLMs, effective text chunking strategies are crucial for managing 
    context windows and ensuring relevant information retention.
    
    Semantic chunking goes beyond simple character-based splitting by 
    analyzing the contextual meaning of text segments. By using 
    embeddings to measure semantic similarity between adjacent text 
    blocks, this approach groups together content that discusses 
    related concepts while separating distinct topics. The result is 
    more coherent and contextually relevant chunks that better preserve 
    the original meaning of the text.
    """
    
    chunks = semantic_split(
        sample_text,
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=92  # Lower = smaller chunks
    )
    
    print(f"\nGenerated {len(chunks)} semantic chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} characters):")
        print("-" * 50)
        print(chunk.strip())
        print("-" * 50)