from sentence_transformers import SentenceTransformer
import chromadb
from src.ingest import read_all_pdfs
from src.chunk import chunk_text


def build_vector_database(pdf_folder="data/pdfs"):
    # --- STEP 1: Read and chunk (reusing Phase 1 & 2 code!) ---
    print("Reading PDFs...")
    docs = read_all_pdfs(pdf_folder)

    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size=500, overlap=50)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "filename": doc["filename"],
                "chunk_index": i,
                "text": chunk
            })

    print(f"{len(all_chunks)} chunks ready.")

    # --- STEP 2: Load the embedding model ---
    # First time: downloads ~80MB. After that: instant.
    print("Loading embedding model")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- STEP 3: Convert chunks to numbers ---
    print("Creating embeddings")
    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(chunk_texts)

    print(f"Created {len(embeddings)} embeddings.")
    print(f"   Each embedding is a list of {len(embeddings[0])} numbers.")

    # --- STEP 4: Store in ChromaDB ---
    print("Storing in ChromaDB...")
    client = chromadb.PersistentClient(path="chroma_data")

    # Delete old data if it exists (so we start fresh each time)
    try:
        client.delete_collection("documind")
    except:
        pass

    collection = client.create_collection("documind")

    # ChromaDB needs a unique ID for each chunk
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    # Metadata = extra info we want to remember alongside each chunk
    metadatas = [
        {"filename": c["filename"], "chunk_index": c["chunk_index"]}
        for c in all_chunks
    ]

    # Store everything
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunk_texts,
        metadatas=metadatas
    )

    print(f"Stored {len(ids)} chunks in ChromaDB!")
    return collection


if __name__ == "__main__":
    build_vector_database()