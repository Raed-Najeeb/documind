import os
from dotenv import load_dotenv
from groq import Groq
from src.retrieve import retrieve_chunks

# Load secret API key from .env file
load_dotenv()

# Initialize the Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(question):
    """
    1. Retrieves the top 3 relevant chunks from our vector database.
    2. Builds a strict, grounded prompt.
    3. Asks the free Llama 3 model on Groq to answer.
    """
    # Step 1: Get the most relevant context chunks
    chunks = retrieve_chunks(question, top_k=3)

    # Step 2: Combine retrieved chunks into a single context string
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i} - File: {chunk['filename']}, Chunk: {chunk['chunk_index']}]\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_blocks)

    # Step 3: Construct the prompt
    system_prompt = (
        "You are DocuMind, an accurate AI research assistant.\n"
        "Your job is to answer the user's question using ONLY the provided sources.\n\n"
        "STRICT RULES:\n"
        "1. Base your answer strictly on the provided sources.\n"
        "2. If the context does not contain enough information to answer the question, say:\n"
        "   'I don't have enough information in the provided documents to answer that.'\n"
        "3. Do NOT make assumptions or hallucinate.\n"
        "4. Always include citations at the end of your answer pointing to the sources used (e.g. [Source 1])."
    )

    user_prompt = f"SOURCES:\n{context_text}\n\nQUESTION: {question}\n\nANSWER:"

    # Step 4: Ask the LLM
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",  # Free, super fast, high quality
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1  # Low temperature = strictly factual, no creative guessing
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Test 1: A question present in your PDF
    q1 = "What is the study plan and what are its core rules?"
    print(f"Question 1: {q1}")
    print("Generating answer...\n")
    print(generate_answer(q1))
    print("\n" + "=" * 60 + "\n")

    # Test 2: A question NOT present in your PDF (tests hallucination prevention!)
    q2 = "What is the capital city of France?"
    print(f"Question 2: {q2}")
    print("Generating answer...\n")
    print(generate_answer(q2))