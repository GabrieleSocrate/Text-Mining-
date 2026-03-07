import os
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


# ----------------------------
# CONFIG
# ----------------------------
CSV_PATH = "dataset_RAG.csv"
TEXT_COL = "description"              # <-- metti qui la colonna giusta
INDEX_DIR = "faiss_index_dataset"     # cartella indice

# Modello embeddings (gratis, locale)
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Modello LLM locale via Ollama
OLLAMA_MODEL = "llama3.1:8b"


# ----------------------------
# LOAD DATASET
# ----------------------------
df = pd.read_csv(CSV_PATH)
texts = df[TEXT_COL].dropna().astype(str).tolist()

documents = [
    Document(page_content=t, metadata={"row_id": i})
    for i, t in enumerate(texts)
]

# Chunking (consigliato se i testi sono lunghi)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# ----------------------------
# EMBEDDINGS (LOCAL)
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    model_kwargs={"device": "cuda"})

# ----------------------------
# FAISS (LOCAL CACHE)
# ----------------------------
# Se cambi EMB_MODEL o chunk params, elimina INDEX_DIR e rigenera.
if os.path.exists(INDEX_DIR):
    vector_store = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_DIR)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# ----------------------------
# LLM (LOCAL via OLLAMA)
# ----------------------------
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

async def rag_answer(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    messages = [
        SystemMessage(content=(
            "You are a helpful assistant. Answer ONLY using the provided context. "
            "If the answer is not in the context, say you don't know."
        )),
        HumanMessage(content=f"Question:\n{question}\n\nContext:\n{context}")
    ]

    return llm.invoke(messages).content
