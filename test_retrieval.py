"""
Script de diagnostic du RAG — à exécuter depuis demo_chatbot/
"""
import os
import sys

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

print("=" * 60)
print("DIAGNOSTIC RAG — MediBot")
print("=" * 60)

# 1. Vérifier ChromaDB
print(f"\n[1] ChromaDB path : {CHROMA_DIR}")
print(f"    Existe        : {os.path.exists(CHROMA_DIR)}")

if not os.path.exists(CHROMA_DIR):
    print("\n❌ PROBLÈME : chroma_db inexistant — index_docs.py n'a jamais été exécuté !")
    print("   Solution   : python index_docs.py")
    sys.exit(1)

import chromadb
client      = chromadb.PersistentClient(path=CHROMA_DIR)
collections = client.list_collections()
print(f"    Collections : {[c.name for c in collections]}")

if not collections:
    print("\n❌ PROBLÈME : ChromaDB vide — aucune collection !")
    print("   Solution   : python index_docs.py")
    sys.exit(1)

col = client.get_collection("medibot_docs")
count = col.count()
print(f"    Chunks      : {count}")

if count == 0:
    print("\n❌ PROBLÈME : collection medibot_docs vide !")
    sys.exit(1)

print(f"\n✅ ChromaDB OK — {count} chunks indexés")

# 2. Test de retrieval direct (sans API Mistral)
print("\n[2] Test retrieval direct (peek des 3 premiers chunks) :")
sample = col.peek(limit=3)
for i, (doc, meta) in enumerate(zip(sample["documents"], sample["metadatas"])):
    print(f"\n  Chunk {i+1} [{meta.get('filename','?')}]:")
    print(f"  {doc[:200]}...")

# 3. Test de retrieval avec embedding Mistral
print("\n[3] Test query avec embedding Mistral :")
TEST_QUESTIONS = [
    "Vaccin pour le Hadj",
    "Carte sanitaire Tunis",
    "Vaccination enfants polio",
]

try:
    from index_docs import query_documents
    for q in TEST_QUESTIONS:
        print(f"\n  Q: '{q}'")
        results = query_documents(q, n_results=3)
        if not results:
            print("  ❌ Aucun résultat retourné !")
        else:
            for r in results:
                print(f"  → [{r['source']}] {r['text'][:150]}...")
except Exception as e:
    print(f"  ❌ ERREUR lors du retrieval : {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Diagnostic terminé.")
