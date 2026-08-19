"""
Test end-to-end : retrieval + Groq response
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medical_bot import get_response
from index_docs import query_documents

TEST_QUESTIONS = [
    "Combien coûte le vaccin contre la méningite pour le Hadj ?",
    "Quels vaccins faut-il faire pour voyager en Arabie Saoudite ?",
    "Où est situé le centre de vaccination internationale à Sfax ?",
]

print("=" * 60)
print("TEST END-TO-END — Retrieval + Groq")
print("=" * 60)

for q in TEST_QUESTIONS:
    print(f"\nQ: {q}")
    print("-" * 40)

    # Afficher les chunks récupérés
    chunks = query_documents(q, n_results=3)
    print(f"  [Retrieval] {len(chunks)} chunks trouvés :")
    for c in chunks:
        print(f"    · [{c['source']}] {c['text'][:100]}...")

    # Réponse finale du bot
    print(f"\n  [Bot Response]:")
    try:
        resp = get_response(q)
        print(f"  {resp[:500]}")
    except Exception as e:
        print(f"  ❌ ERREUR : {e}")

print("\n" + "=" * 60)
