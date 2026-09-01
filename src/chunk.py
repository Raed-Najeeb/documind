from src.ingest import read_all_pdfs


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Cuts a long text into smaller pieces.

    chunk_size = how many characters per slice (like bread slice thickness)
    overlap = how many characters to repeat between slices (shared crust)
    """
    chunks = []
    start = 0

    while start < len(text):
        # Cut a slice from 'start' to 'start + chunk_size'
        end = start + chunk_size
        slice_of_text = text[start:end]

        # Only keep it if it's not empty
        if slice_of_text.strip():
            chunks.append(slice_of_text)

        # Move forward, but step back a little for overlap
        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    # Step 1: Read the PDFs (reusing your Phase 1 code!)
    docs = read_all_pdfs("data/pdfs")

    if not docs:
        print("No documents found. Make sure you have a PDF in data/pdfs/")
    else:
        all_chunks = []

        for doc in docs:
            chunks = chunk_text(doc["text"], chunk_size=500, overlap=50)

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "filename": doc["filename"],
                    "chunk_index": i,
                    "text": chunk
                })

        print(f"\n Total chunks created: {len(all_chunks)}")
        print(f"   From {len(docs)} document(s)")

        # Show the first 3 chunks so you can see the overlap
        print("\n--- First 3 chunks ---")
        for i in range(min(3, len(all_chunks))):
            c = all_chunks[i]
            print(f"\n[Chunk {c['chunk_index']}] ({len(c['text'])} chars)")
            print(c["text"][:150] + "...")