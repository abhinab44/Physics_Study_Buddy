# Retrieval Validation
# Domain: Study Buddy — Physics

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from study_buddy.knowledge_base import get_collection, get_embedder

def test_query(collection, embedder, query, expected_topic):
    print(f"\nQuery: {query}")
    print(f"Expected topic: {expected_topic}")

    q_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)

    topics = [m["topic"] for m in results["metadatas"][0]]
    texts = results["documents"][0]

    print(f"Top 3 results:")
    for i, (topic, text) in enumerate(zip(topics, texts)):
        print(f"  {i+1}. [{topic}] {text[:100]}...")

    if expected_topic in topics:
        print(f"PASS: '{expected_topic}' found in top 3")
        return True
    else:
        print(f"FAIL: '{expected_topic}' NOT in top 3")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Physics Knowledge Base - Retrieval Validation")
    print("=" * 50)

    collection = get_collection()
    embedder = get_embedder()
    print(f"Total documents: {collection.count()}")

    tests = [
        (
            "What is Newton's second law of motion?",
            "Laws of Motion"
        ),
        (
            "Explain Snell's law of refraction",
            "Ray Optics"
        ),
        (
            "What is Heisenberg's uncertainty principle?",
            "Quantum Mechanics"
        ),
        (
            "What is Gauss's theorem in electrostatics?",
            "Electrostatics"
        ),
        (
            "Explain the Carnot cycle and its efficiency",
            "Heat and Thermodynamics"
        ),
    ]

    passed = 0
    for query, expected in tests:
        if test_query(collection, embedder, query, expected):
            passed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{len(tests)} passed")
    print("=" * 50)
