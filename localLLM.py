# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load the model and tokenizer from Hugging Face
# model_name = "Qwen/Qwen1.5-0.5B-Chat"  # Chat model, not for classification
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Save locally (optional)
# model.save_pretrained("./my_local_model")
# tokenizer.save_pretrained("./my_local_model")

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from transformers import pipeline

# 1. Prepare docs & FAISS index (same as before)…
texts = [
    "The sun rises in the east.",
    "Python is a popular programming language.",
    "Delhi is the capital of India.",
    "OpenAI developed ChatGPT.",
    "Heart of India Delhi",
]
docs = [Document(page_content=t) for t in texts]
emb = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
vs = FAISS.from_documents(docs, emb)
vs.save_local("faiss_Vectorstore")
vs = FAISS.load_local("faiss_Vectorstore", emb, allow_dangerous_deserialization=True)
retriever = vs.as_retriever(search_kwargs={"k": 2})

# 2. Wrap your local text-generation pipeline
pipe = pipeline(
    "text-generation",
    model="./my_local_model",
    tokenizer="./my_local_model",
    device="cpu",
    framework="pt"
)
llm = HuggingFacePipeline(
    pipeline=pipe,
    model_kwargs={
        "max_new_tokens": 50,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9
    }
)

# 3. Create a PromptTemplate instead of a raw string
template = """Use the following context to answer the question.

{context}

Question: {question}

Answer:"""
prompt_template = PromptTemplate.from_template(template)

# 4. Build RetrievalQA with the PromptTemplate
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": prompt_template
    }
)

# 5. Run a query
result = qa.invoke({"query": "What is the capital of India?"})
print("► Answer:\n", result["result"])
print("\n► Sources:")
for doc in result["source_documents"]:
    print("-", doc.page_content)
