import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Directory where ChromaDB saves data locally
CHROMA_DIR = "chroma_db/"

def get_embeddings():
    """
    OpenAI embeddings — converts text into numbers (vectors).
    text-embedding-3-small is cheap and very accurate.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )


def build_vector_store(chunks: list):
    """
    Takes chunks from loader.py and stores them in ChromaDB.
    This only needs to run ONCE per document set.
    """
    print("Building vector store...")

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print(f"Vector store built with {len(chunks)} chunks")
    print(f"Saved to: {CHROMA_DIR}")
    return vector_store


def load_vector_store():
    """
    Loads existing ChromaDB from disk.
    Use this after the first time you've already built it.
    """
    print("Loading existing vector store...")

    embeddings = get_embeddings()

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    print("Vector store loaded successfully")
    return vector_store


def get_retriever(vector_store, k: int = 5):
    """
    Creates a retriever — used to search the vector store.
    k = number of chunks to return per query
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def vector_store_exists():
    """
    Check if vector store already exists on disk.
    Prevents rebuilding every time app starts.
    """
    return os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0


# Test this file directly
if __name__ == "__main__":
    from ingestion.loader import load_and_split

    chunks = load_and_split("data/")

    if chunks:
        store = build_vector_store(chunks)
        retriever = get_retriever(store)
        results = retriever.invoke("test query")
        print(f"\nTest search returned {len(results)} results")
    else:
        print("Add PDF files to data/ folder first, then run again.")