import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
import chromadb

# Global cache so we don't reload heavy models on every search
_embed_model = None
_reranker_model = None
_bm25_index = None
_all_documents = None
_all_metadatas = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def get_reranker():
    global _reranker_model
    if _reranker_model is None:
        # A small, super-accurate judge model
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model


def get_bm25_and_docs():
    """Fetches all chunks from ChromaDB and creates the BM25 keyword index."""
    global _bm25_index, _all_documents, _all_metadatas
    if _bm25_index is None:
        client = chromadb.PersistentClient(path="chroma_data")
        collection = client.get_collection("documind")
        data = collection.get()

        _all_documents = data["documents"]
        _all_metadatas = data["metadatas"]

        # BM25 tokenizes text into lowercase words
        tokenized_corpus = [doc.lower().split() for doc in _all_documents]
        _bm25_index = BM25Okapi(tokenized_corpus)

    return _bm25_index, _all_documents, _all_metadatas


def reciprocal_rank_fusion(vector_indices, bm25_indices, k=60):
    """
    Combines ranks from Vector Detective and BM25 Detective.
    Each chunk gets points: 1 / (60 + rank).
    """
    scores = {}

    for rank, idx in enumerate(vector_indices):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    for rank, idx in enumerate(bm25_indices):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    # Sort candidates by combined score (highest first)
    sorted_candidates = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_candidates


def retrieve_chunks(question, top_k=3, candidate_pool_size=10):
    """
    1. Vector search grabs top candidates by meaning.
    2. BM25 search grabs top candidates by exact keywords.
    3. RRF combines both lists.
    4. Cross-Encoder re-ranks the top merged candidates to pick the final winners.
    """
    client = chromadb.PersistentClient(path="chroma_data")
    collection = client.get_collection("documind")

    # --- 1. VECTOR SEARCH ---
    embed_model = get_embed_model()
    q_emb = embed_model.encode([question]).tolist()
    vector_res = collection.query(query_embeddings=q_emb, n_results=candidate_pool_size)

    # Convert chunk_id (e.g. "chunk_3") into integer index 3
    vector_indices = [int(cid.split("_")[1]) for cid in vector_res["ids"][0]]

    # --- 2. BM25 KEYWORD SEARCH ---
    bm25, docs, metadatas = get_bm25_and_docs()
    query_tokens = question.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_pool_size].tolist()

    # --- 3. MERGE WITH RRF ---
    merged_indices = reciprocal_rank_fusion(vector_indices, bm25_indices, k=60)
    top_candidates = merged_indices[:candidate_pool_size]

    # --- 4. RE-RANK WITH CROSS-ENCODER ---
    reranker = get_reranker()
    candidate_texts = [docs[idx] for idx in top_candidates]

    # Pair the question with each candidate chunk
    pairs = [[question, text] for text in candidate_texts]
    rerank_scores = reranker.predict(pairs)

    # Rank the candidate chunks by re-ranker score
    scored_candidates = []
    for i, idx in enumerate(top_candidates):
        scored_candidates.append({
            "text": docs[idx],
            "filename": metadatas[idx]["filename"],
            "chunk_index": metadatas[idx]["chunk_index"],
            "score": float(rerank_scores[i])
        })

    # Sort descending by judge's score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    return scored_candidates[:top_k]


if __name__ == "__main__":
    test_q = "What is Section 26 about and what does it promise?"
    print(f"❓ Testing Hybrid Retrieval + Re-ranking for: '{test_q}'\n")

    results = retrieve_chunks(test_q, top_k=3)

    for rank, res in enumerate(results, 1):
        print(f"--- Rank {rank} (Re-ranker Score: {res['score']:.4f}) ---")
        print(f"Source: {res['filename']} (Chunk {res['chunk_index']})")
        print(f"Text:\n{res['text']}\n")