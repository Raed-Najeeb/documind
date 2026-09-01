import json
import os
from dotenv import load_dotenv
from groq import Groq
from src.retrieve import retrieve_chunks
from src.generate import generate_answer

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def judge_faithfulness(question, context, answer):
    """
    Grades: Is the answer grounded in the retrieved context?
    Returns a score from 1 to 5.
    """
    prompt = f"""You are a strict grading judge. Score the FAITHFULNESS of the answer.

FAITHFULNESS means: Is every claim in the answer directly supported by the provided context?

SCORING:
1 = Completely hallucinated, nothing in the answer is in the context
2 = Mostly made up, only tiny details match the context
3 = Half grounded, half invented
4 = Mostly grounded, minor unsupported details
5 = Every single claim is directly supported by the context

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

Respond with ONLY a single number (1-5). Nothing else."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    score_text = response.choices[0].message.content.strip()
    # Extract just the number
    for char in score_text:
        if char.isdigit():
            return int(char)
    return 1


def judge_relevance(question, answer):
    """
    Grades: Does the answer actually address the question?
    Returns a score from 1 to 5.
    """
    prompt = f"""You are a strict grading judge. Score the RELEVANCE of the answer.

RELEVANCE means: Does the answer directly address what the user asked?

SCORING:
1 = Completely off-topic, does not address the question at all
2 = Barely related, mostly talks about something else
3 = Partially addresses the question but wanders off
4 = Mostly on-topic, addresses the question with minor tangents
5 = Directly and precisely answers the question

QUESTION: {question}

ANSWER: {answer}

Respond with ONLY a single number (1-5). Nothing else."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    score_text = response.choices[0].message.content.strip()
    for char in score_text:
        if char.isdigit():
            return int(char)
    return 1


def run_evaluation():
    # Load test questions
    with open("data/eval_dataset.json", "r") as f:
        test_set = json.load(f)

    print(f"Loaded {len(test_set)} test questions.\n")
    print("=" * 70)

    results = []

    for i, test in enumerate(test_set, 1):
        question = test["question"]
        expected = test["expected_answer"]

        print(f"\nTest {i}/{len(test_set)}: {question}")

        # 1. Get retrieved chunks (for context)
        chunks = retrieve_chunks(question, top_k=3)
        context = "\n\n".join([c["text"] for c in chunks])

        # 2. Generate RAG answer
        answer = generate_answer(question)

        # 3. Judge Faithfulness
        faith_score = judge_faithfulness(question, context, answer)

        # 4. Judge Relevance
        rel_score = judge_relevance(question, answer)

        results.append({
            "question": question,
            "expected": expected,
            "answer": answer,
            "faithfulness": faith_score,
            "relevance": rel_score
        })

        print(f"   Faithfulness: {faith_score}/5 | Relevance: {rel_score}/5")

    # --- SUMMARY ---
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel = sum(r["relevance"] for r in results) / len(results)

    print("\n" + "=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)
    print(f"   Average Faithfulness: {avg_faith:.1f}/5  ({avg_faith/5*100:.0f}%)")
    print(f"   Average Relevance:    {avg_rel:.1f}/5  ({avg_rel/5*100:.0f}%)")
    print(f"   Total Questions:      {len(results)}")
    print("=" * 70)

    # Save results to file
    with open("data/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDetailed results saved to data/eval_results.json")


if __name__ == "__main__":
    run_evaluation()