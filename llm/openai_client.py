import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

load_dotenv()

# Initialize OpenAI client once
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model to use
MODEL = "gpt-4o"

# System prompt — this controls how GPT-4o behaves
SYSTEM_PROMPT = """You are an expert research assistant. Your job is to answer 
questions based ONLY on the provided document context below.

Rules you must follow:
1. Answer ONLY from the context provided — never from general knowledge
2. Always cite your source at the end of each claim like this: [source: filename, page X]
3. If the answer is not in the context, say: "I could not find this information in the provided documents."
4. Be concise and clear
5. If multiple sources support the answer, cite all of them
"""


def format_context(documents: list) -> str:
    """
    Formats retrieved document chunks into a clean context string
    that gets sent to GPT-4o.

    Each chunk is formatted like:
    ---
    Source: report.pdf (page 3)
    Content: The infrastructure costs rose by 23%...
    ---

    This makes it easy for GPT-4o to cite sources accurately.
    """
    context_parts = []

    for i, doc in enumerate(documents):
        # Extract source info from metadata
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        score = doc.metadata.get("rerank_score", "N/A")

        # Format each chunk clearly
        chunk_text = f"""
[Chunk {i+1}]
Source: {os.path.basename(source)} | Page: {page} | Relevance: {score}
Content: {doc.page_content}
"""
        context_parts.append(chunk_text)

    return "\n---\n".join(context_parts)


def count_tokens(text: str, model: str = MODEL) -> int:
    """
    Counts how many tokens a text uses.
    Important for cost tracking — OpenAI charges per token.
    1 token ≈ 4 characters or 0.75 words.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate
        return len(text) // 4


def get_answer(query: str, documents: list) -> dict:
    """
    Main function — sends question + context to GPT-4o
    and returns the answer.

    Returns a dict with:
    - answer: the text response
    - input_tokens: tokens used in the prompt (for cost tracking)
    - output_tokens: tokens used in the response
    - total_cost: estimated cost in USD
    """
    # Format retrieved chunks into context
    context = format_context(documents)

    # Build the full prompt
    user_message = f"""Here is the context from the documents:

{context}

Question: {query}

Please answer based only on the context above."""

    # Count input tokens for cost tracking
    input_tokens = count_tokens(SYSTEM_PROMPT + user_message)

    print(f"Sending to GPT-4o... ({input_tokens} input tokens)")

    # Call OpenAI API
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1,  # Low temperature = more factual, less creative
        max_tokens=1000
    )

    # Extract answer
    answer = response.choices[0].message.content
    output_tokens = response.usage.completion_tokens
    input_tokens = response.usage.prompt_tokens

    # Calculate cost (GPT-4o pricing as of 2024)
    # Input: $5 per 1M tokens, Output: $15 per 1M tokens
    input_cost = (input_tokens / 1_000_000) * 5
    output_cost = (output_tokens / 1_000_000) * 15
    total_cost = round(input_cost + output_cost, 6)

    print(f"Answer received! ({output_tokens} output tokens, ${total_cost} cost)")

    return {
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost": total_cost
    }


def get_answer_streaming(query: str, documents: list):
    """
    Streaming version — yields answer tokens one by one.
    This creates the 'typing' effect in the UI.

    Usage:
        for token in get_answer_streaming(query, docs):
            print(token, end="", flush=True)
    """
    context = format_context(documents)

    user_message = f"""Here is the context from the documents:

{context}

Question: {query}

Please answer based only on the context above."""

    print(f"Streaming answer from GPT-4o...")

    # Stream the response
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1,
        max_tokens=1000,
        stream=True  # This enables streaming!
    )

    # Yield each token as it arrives
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


# Test this file directly
if __name__ == "__main__":
    from langchain_core.documents import Document

    # Create fake test documents to test without real PDFs
    test_docs = [
        Document(
            page_content="Infrastructure costs rose by 23% in Q3 2024, driven by cloud compute expenses.",
            metadata={"source": "report_2024.pdf", "page": 14, "rerank_score": 0.92}
        ),
        Document(
            page_content="The report recommends hybrid deployment strategies to reduce overhead by up to 40%.",
            metadata={"source": "report_2024.pdf", "page": 21, "rerank_score": 0.87}
        )
    ]

    query = "What happened to infrastructure costs?"

    print("--- Testing normal response ---")
    result = get_answer(query, test_docs)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nTokens used: {result['input_tokens']} in, {result['output_tokens']} out")
    print(f"Cost: ${result['total_cost']}")

    print("\n--- Testing streaming response ---")
    for token in get_answer_streaming(query, test_docs):
        print(token, end="", flush=True)
    print("\nStreaming done!")