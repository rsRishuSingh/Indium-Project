import os
import fitz  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from groq import Groq

# Initialize Groq client for LLM 
hf_api_key = os.environ.get("GROQ_API_KEY")
groq_api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Load embedding model from Hugging Face
embeddings_model  = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B"
)
def useGroq():
    client = Groq(groq_api_key)
    completion = client.chat.completions.create(
        model="qwen-qwq-32b",
        messages=[
        {
            "role": "user",
            "content": ""
        }
        ],
        temperature=0.6,
        max_completion_tokens=4096,
        top_p=0.95,
        stream=True,
        stop=None,
    )
    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")


# spiltter is chunker here  
# plitter.split_text(text) generate txtual chunks
# Define splitting functions 

def character_split(text, chunk_size=1000, chunk_overlap=200):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def recursive_split(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def semantic_split(text, embeddings_model , **kwargs):
    chunker = SemanticChunker(embeddings_model, **kwargs)
    print(chunker)
    chunks = chunker.split_text(text)
    return chunks

def page_wise_split(page_texts):
    return page_texts

def print_chunks(chunks, num_to_print=3):
    print(f"Total chunks: {len(chunks)}")
    # for i in range(min(num_to_print, len(chunks))):
    #     chunk = chunks[i]
    #     preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
    #     print(f"Chunk {i+1} (length {len(chunk)}): {preview}")
    for i in range(len(chunks)):
   
        print(f"Chunk {i+1}-> {chunks[i]}", end='\v')

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    page_texts = []
    full_text_list = []
    for page in doc:
        text = page.get_text("text")
        page_texts.append(text)
        full_text_list.append(text);
    doc.close()
    full_text = "\n".join(page_texts)
    return full_text, page_texts, full_text_list

# Main processing
INP_DIR = 'PDFs/'
BASE_EXT = '.pdf'

def main():

    pdf_files = ["TESLA"]  
    
    for pdf in pdf_files:
        try:
            # Extract text from PDF
            
            pdf_path = INP_DIR + pdf + BASE_EXT
            full_text, page_texts,full_text_list  = extract_text_from_pdf(pdf_path)
            
            # Apply splitting methods

            # chunks_char = character_split(page_texts[0])
            # chunks_recursive = recursive_split(full_text)
            chunks_semantic = semantic_split(page_texts[0],embeddings_model,breakpoint_threshold_type="percentile")
            # chunks_page = page_wise_split(page_texts)
            
            # Print results

            print(f"\nProcessing {pdf}")
            # print("### Character Split:")
            # print_chunks(chunks_char)
            # print("\n### Recursive Split:")
            # print_chunks(chunks_recursive)
            print("\n### Semantic Split:")
            print_chunks(chunks_semantic)
            # print("\n### Page-wise Split:")
            # print_chunks(chunks_page)
        
        except Exception as e:
            print(f"Error processing {pdf}: {e}")




main()