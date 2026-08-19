"""
index_docs.py
-------------
Indexation des documents de data_chatbot/ dans ChromaDB.
Utilise l'API Mistral (mistral-embed) pour les embeddings —
zéro mémoire locale, multilingue français/arabe.

Usage (une seule fois) :
    python index_docs.py
"""

import os
import time
import chromadb
from pypdf import PdfReader
from mistralai.client import Mistral

# ── Client Mistral ─────────────────────────────────────────────────────────────
def _load_mistral_key() -> str:
    key = os.environ.get("MISTRAL_API_KEY", "")
    if key:
        return key
    try:
        import yaml
        _here = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(_here, "..", "configs", "medical_bot.yaml")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("chatbot", {}).get("mistral_api_key", "")
    except Exception:
        return ""

mistral_client = Mistral(api_key=_load_mistral_key())
EMBED_MODEL    = "mistral-embed"


def get_embedding(text: str) -> list:
    """Embedding d'un seul texte via Mistral API."""
    response = mistral_client.embeddings.create(model=EMBED_MODEL, inputs=[text])
    return response.data[0].embedding


def get_embeddings_batch(texts: list) -> list:
    """Embedding d'un batch de textes via Mistral API."""
    response = mistral_client.embeddings.create(model=EMBED_MODEL, inputs=texts)
    return [item.embedding for item in response.data]


# ── ChromaDB : base locale persistante ────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection    = chroma_client.get_or_create_collection(name="medibot_docs")


# ── Chargement des documents ───────────────────────────────────────────────────

def load_documents_from_directory(directory_path: str) -> list:
    documents   = []
    folder_name = os.path.basename(directory_path)

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        text = ""

        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        elif filename.endswith(".pdf"):
            reader = PdfReader(filepath)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

        if text.strip():
            documents.append({"text": text, "source_dir": folder_name, "filename": filename})
            print(f"  ✓ Chargé : {folder_name}/{filename} ({len(text)} chars)")

    return documents


# ── Découpage en chunks ────────────────────────────────────────────────────────

def split_text(text: str, chunk_size: int = 600, chunk_overlap: int = 150) -> list:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - chunk_overlap
    return chunks


# ── Pipeline principal ─────────────────────────────────────────────────────────

DATA_DIRS = [
    "data_chatbot/carte_sanitaaire",
    "data_chatbot/charte_patient",
    "data_chatbot/sante_voyageur",
    "data_chatbot/vaccination_enfants",
]


def run_indexing():
    """Lance l'indexation complète."""
    global collection

    try:
        chroma_client.delete_collection(name="medibot_docs")
        print("[index_docs] Collection existante supprimée.")
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(name="medibot_docs")

    print("=== Chargement des documents ===")
    all_documents = []
    for folder in DATA_DIRS:
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"  ⚠ Dossier introuvable : {folder_path}")
            continue
        all_documents.extend(load_documents_from_directory(folder_path))
    print(f"\nTotal documents : {len(all_documents)}")

    print("\n=== Découpage en chunks ===")
    all_chunks, all_ids, all_metadatas = [], [], []
    chunk_counter = 0
    for doc in all_documents:
        for chunk in split_text(doc["text"]):
            if chunk.strip():
                all_chunks.append(chunk)
                all_ids.append(f"chunk_{chunk_counter}")
                all_metadatas.append({"source_dir": doc["source_dir"], "filename": doc["filename"]})
                chunk_counter += 1
    print(f"Total chunks : {len(all_chunks)}")

    print("\n=== Calcul des embeddings via Mistral API ===")
    BATCH_SIZE     = 20
    embeds_to_add  = []
    for b in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[b:b + BATCH_SIZE]
        embeds_to_add.extend(get_embeddings_batch(batch))
        print(f"  {min(b + BATCH_SIZE, len(all_chunks))}/{len(all_chunks)}")
        time.sleep(0.3)

    collection.add(
        documents=all_chunks,
        embeddings=embeds_to_add,
        ids=all_ids,
        metadatas=all_metadatas,
    )
    print(f"\n✅ {len(all_chunks)} chunks indexés dans ChromaDB.")


# ── Retrieval ──────────────────────────────────────────────────────────────────

def query_documents(question: str, n_results: int = 5, distance_threshold: float = 1.5) -> list:
    """Recherche les chunks les plus pertinents pour une question.

    Args:
        question: La question à rechercher.
        n_results: Nombre de chunks à récupérer (augmenté à 5 par défaut).
        distance_threshold: Seuil de distance L2 — chunks au-delà sont filtrés
                            (1.5 = permissif, couvre bien les documents techniques/tabulaires).
    """
    query_embedding = get_embedding(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]  # L2 : plus petit = plus similaire

    filtered = []
    for chunk, meta, dist in zip(chunks, metadatas, distances):
        if dist <= distance_threshold:
            filtered.append({
                "text": chunk,
                "source": f"{meta['source_dir']}/{meta['filename']}",
                "distance": round(dist, 4),
            })

    return filtered


if __name__ == "__main__":
    run_indexing()
    print("\n=== Test de retrieval ===")
    results = query_documents("Combien coûte le vaccin contre la méningite pour le Hadj ?", n_results=3)
    for r in results:
        print(f"\n[Source: {r['source']} | distance: {r['distance']}]")
        print(r["text"][:300])
