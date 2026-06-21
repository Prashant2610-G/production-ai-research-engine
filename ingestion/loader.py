import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(data_dir: str = "data/") -> list:
    """
    Load all PDF and text files from the data directory.
    Returns a list of raw documents.
    """
    documents = []

    # Check if directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory at: {data_dir}")
        return documents

    # Load PDF files
    pdf_loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    # Load text files
    txt_loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )

    try:
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
        print(f"Loaded {len(pdf_docs)} PDF pages")
    except Exception as e:
        print(f"Error loading PDFs: {e}")

    try:
        txt_docs = txt_loader.load()
        documents.extend(txt_docs)
        print(f"Loaded {len(txt_docs)} text files")
    except Exception as e:
        print(f"Error loading text files: {e}")

    return documents


def split_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Split documents into smaller chunks.
    chunk_size = how many characters per chunk
    chunk_overlap = how many characters overlap between chunks (prevents losing meaning)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def load_and_split(data_dir: str = "data/") -> list:
    """
    Main function — loads documents and splits them in one step.
    This is what other files will call.
    """
    print(f"Loading documents from: {data_dir}")
    documents = load_documents(data_dir)

    if not documents:
        print("No documents found. Please add PDF or text files to the data/ folder.")
        return []

    chunks = split_documents(documents)
    print(f"Ready: {len(chunks)} chunks prepared for indexing")
    return chunks


# Test the loader directly
if __name__ == "__main__":
    chunks = load_and_split("data/")
    if chunks:
        print("\n--- Sample Chunk ---")
        print(f"Content: {chunks[0].page_content[:200]}...")
        print(f"Source: {chunks[0].metadata}")