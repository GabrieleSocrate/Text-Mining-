import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama

# ---------------------------
# Configurations
# ---------------------------
CSV_PATH   = "dataset_RAG.csv"
INDEX_PATH = "faiss_index.bin"
DOCS_PATH  = "faiss_docs.npy"
EMB_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.1:8b"
# OLLAMA_MODEL = "qwen2:1.5b"
TOP_K = 20

# ----------------------------
# LOAD & CHUNK DATASET
# ----------------------------
df = pd.read_csv(CSV_PATH)
texts = df.dropna().astype(str).values.flatten().tolist() # In this way we have a list of all documents 

documents = [
    Document(page_content=t, metadata={"row_id": i})
    for i, t in enumerate(texts)
]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
chunk_texts = [c.page_content for c in chunks]

# ----------------------------
# EMBEDDINGS
# ----------------------------
embedding_model = SentenceTransformer(EMB_MODEL, device="cuda")
embedding_dim   = embedding_model.get_sentence_embedding_dimension()

# ----------------------------
# FAISS (Create or Load Document Embeddings)
# ----------------------------
if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
    index = faiss.read_index(INDEX_PATH)
    chunk_texts_arr = np.load(DOCS_PATH, allow_pickle=True).tolist()
else:
    doc_embeddings = embedding_model.encode(chunk_texts)
    doc_embeddings = np.array(doc_embeddings, dtype="float32")
    faiss.normalize_L2(doc_embeddings)

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(doc_embeddings)

    faiss.write_index(index, INDEX_PATH)
    np.save(DOCS_PATH, np.array(chunk_texts, dtype=object))
    chunk_texts_arr = chunk_texts

# ----------------------------
# LLM
# ----------------------------
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

# ----------------------------
# RETRIEVAL FUNCTION
# ----------------------------
def retrieve_documents(question, k = TOP_K):
    query_emb = np.array(embedding_model.encode([question]), dtype="float32")
    faiss.normalize_L2(query_emb)

    distances, indices = index.search(query_emb, k)
    similarities = 1 - distances[0]

    return [
        {"text": chunk_texts_arr[idx], "score": float(score)}
        for idx, score in zip(indices[0], similarities)
    ]

# ----------------------------
# Building the Augmented Prompt
# ----------------------------
def create_augmented_prompt(question, retrieved_docs):
    context = "\n\n---\n\n".join(doc["text"] for doc in retrieved_docs)

    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer ONLY based on the context. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Answer:"
    )

# ----------------------------
# Rag function
# ----------------------------
def rag_answer(question):
    retrieved_docs   = retrieve_documents(question)
    augmented_prompt = create_augmented_prompt(question, retrieved_docs)
    return (llm.invoke(augmented_prompt)).content
