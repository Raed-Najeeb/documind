from sentence_transformers import SentenceTransformer
import chromadb


def retrieve_chunks(question, top_k=3):
    """
    1. Loads the embedding model and the saved ChromaDB filing cabinet.
    2. Turns the question into numbers (embedding).
    3. Finds and returns the top_k closest chunks.
    """
    # Load the same embedding model we used to store the chunks
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Connect to our existing ChromaDB database folder
    client = chromadb.PersistentClient(path="chroma_data")
    collection = client.get_collection("documind")

    # Turn the user's question into numbers
    question_embedding = model.encode([question])

    # Ask ChromaDB: "Which chunks are closest in meaning to these numbers?"
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=top_k
    )

    # Format the results into clean dictionaries
    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": results["distances"][0][i]  # Lower number = closer match
        })

    return retrieved


if __name__ == "__main__":
    # Let's test it with a question about your PDF!
    test_question = "What is the study plan and how does it work?"

    print(f"Question: {test_question}\n")
    print("Searching ChromaDB for closest chunks...\n")

    chunks = retrieve_chunks(test_question, top_k=3)

    for rank, chunk in enumerate(chunks, 1):
        print(f"--- Match #{rank} (from {chunk['filename']}, Chunk {chunk['chunk_index']}) ---")
        print(f"Distance Score: {chunk['distance']:.4f} (lower is better)")
        print(f"Content:\n{chunk['text']}")
        print("-" * 60 + "\n")