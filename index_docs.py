"""
index_docs.py
-------------
Script à lancer UNE SEULE FOIS pour indexer les documents de data_chatbot/
dans ChromaDB avec des embeddings multilingues (français/arabe).

Usage :
    python demo_chatbot/index_docs.py
"""

import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ── Modèle multilingue LÉGER (~90MB en mémoire) ───────────────────────────────
# paraphrase-multilingual-MiniLM-L12-v2 = 118MB disque, ~200MB RAM
# Bien plus petit que le modèle complet, supporte français/arabe
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def get_embedding(text: str) -> list:
    return model.encode(text, convert_to_numpy=True).tolist()


# ── ChromaDB : base locale persistante ────────────────────────────────────────
# Le dossier chroma_db/ sera créé dans demo_chatbot/
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

chroma_client   = chromadb.PersistentClient(path=CHROMA_DIR)
collection      = chroma_client.get_or_create_collection(name="medibot_docs")


# ── Chargement des documents ──────────────────────────────────────────────────

def load_documents_from_directory(directory_path: str) -> list:
    """
    Lit tous les fichiers .txt et .pdf dans un dossier.
    Retourne une liste de dicts : {id, text, source_dir, filename}
    """
    documents = []
    folder_name = os.path.basename(directory_path)

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        text = ""

        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

        elif filename.endswith(".pdf"):
            reader = PdfReader(filepath)
            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        if text.strip():
            documents.append({
                "id":         filename,
                "text":       text,
                "source_dir": folder_name,   # ex: "vaccination_enfants"
                "filename":   filename,
            })
            print(f"  ✓ Chargé : {folder_name}/{filename} ({len(text)} chars)")

    return documents


# ── Découpage en chunks ───────────────────────────────────────────────────────

def split_text(text: str, chunk_size: int = 300, chunk_overlap: int = 100) -> list:
    """
    Découpe un texte en chunks de taille fixe avec overlap.
    chunk_overlap augmenté à 100 pour ne pas couper les phrases importantes.
    """
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# ── Pipeline principal ────────────────────────────────────────────────────────

DATA_DIRS = [
    "data_chatbot/carte_sanitaaire",
    "data_chatbot/charte_patient",
    "data_chatbot/sante_voyageur",
    "data_chatbot/vaccination_enfants",
]


def run_indexing():
    """Lance l'indexation complète — à appeler uniquement depuis __main__."""
    # Supprimer et recréer la collection pour repartir propre
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
        docs = load_documents_from_directory(folder_path)
        all_documents.extend(docs)

    print(f"\nTotal documents chargés : {len(all_documents)}")

    print("\n=== Découpage en chunks ===")
    all_chunks    = []
    all_ids       = []
    all_metadatas = []

    chunk_counter = 0
    for doc in all_documents:
        chunks = split_text(doc["text"], chunk_size=1000, chunk_overlap=100)
        for chunk in chunks:
            if not chunk.strip():
                continue
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_counter}")
            all_metadatas.append({
                "source_dir": doc["source_dir"],
                "filename":   doc["filename"],
            })
            chunk_counter += 1

    print(f"Total chunks créés : {len(all_chunks)}")

    print("\n=== Calcul des embeddings et indexation ===")
    try:
        existing_ids = set(collection.get(ids=all_ids).get("ids", []))
    except Exception:
        existing_ids = set()

    missing = [(i, _id) for i, _id in enumerate(all_ids) if _id not in existing_ids]
    print(f"Chunks à ajouter : {len(missing)} (déjà indexés : {len(existing_ids)})")

    if missing:
        indices      = [i for i, _ in missing]
        docs_to_add  = [all_chunks[i]    for i in indices]
        ids_to_add   = [all_ids[i]       for i in indices]
        metas_to_add = [all_metadatas[i] for i in indices]

        BATCH_SIZE    = 32
        embeds_to_add = []
        for b in range(0, len(docs_to_add), BATCH_SIZE):
            batch = docs_to_add[b:b + BATCH_SIZE]
            embeds_to_add.extend(model.encode(batch, convert_to_numpy=True).tolist())
            print(f"  Embeddings calculés : {min(b + BATCH_SIZE, len(docs_to_add))}/{len(docs_to_add)}")

        collection.add(
            documents=docs_to_add,
            embeddings=embeds_to_add,
            ids=ids_to_add,
            metadatas=metas_to_add,
        )
        print(f"\n✅ {len(ids_to_add)} chunks ajoutés dans ChromaDB ({CHROMA_DIR})")
    else:
        print("✅ Collection déjà à jour, rien à ajouter.")

# ── Fonction de retrieval (importable par medical_bot.py) ─────────────────────

def query_documents(question: str, n_results: int = 3) -> list:
    """
    Recherche les chunks les plus pertinents pour une question.
    Retourne une liste de dicts : {text, source}
    """
    query_embedding = get_embedding(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas","distances"],
    )
    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]

    return [
        {"text": chunk, "source": f"{meta['source_dir']}/{meta['filename']}"}
        for chunk, meta in zip(chunks, metadatas)
    ]


if __name__ == "__main__":
    run_indexing()
    print("\n=== Test de retrieval ===")
    test_q = "prix vaccin méningite ACYW135 Hadj Omra 130 DT"
    results = query_documents(test_q, n_results=3)
    for r in results:
        print(f"\n[Source: {r['source']}]")
        print(r["text"][:300])
