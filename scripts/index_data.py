"""
Database Indexing Script.
Loads mock protocols, computes embeddings, and stores them in persistent ChromaDB.
"""
import os
import json
import logging
from sentence_transformers import SentenceTransformer
import chromadb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IndexData")


def index_protocols():
    """
    Read protocols from JSON, compute vector embeddings using all-MiniLM-L6-v2,
    and persist them inside the ChromaDB collection 'medical_protocols'.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "data", "Medical_Protocols.json")
    db_path = os.path.join(base_dir, "chroma_db")

    logger.info("Loading protocols from: %s", json_path)
    if not os.path.exists(json_path):
        logger.error("JSON data file not found at %s!", json_path)
        return

    with open(json_path, "r", encoding="utf-8") as f:
        protocols = json.load(f)

    logger.info("Initializing ChromaDB PersistentClient at: %s", db_path)
    client = chromadb.PersistentClient(path=db_path)

    logger.info("Getting or creating collection 'medical_protocols'...")
    collection = client.get_or_create_collection(
        name="medical_protocols",
        metadata={"hnsw:space": "cosine"}
    )

    logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for protocol in protocols:
        doc_id = protocol["id"]
        medication = protocol["medication"]
        protocol_type = protocol["protocol_type"]
        department = protocol["department"]
        content = protocol["content"]
        last_updated = protocol["last_updated"]

        priority = "HIGH" if protocol_type == "Override" else "NORMAL"

        existing = collection.get(ids=[doc_id])
        if existing and existing.get("ids"):
            logger.info("Protocol ID '%s' already indexed. Skipping.", doc_id)
            continue

        logger.info("Computing embedding and storing protocol: %s (%s)", medication, protocol_type)
        embedding = model.encode(content).tolist()

        metadata = {
            "id": doc_id,
            "medication": medication,
            "department": department,
            "protocol_type": protocol_type,
            "priority": priority,
            "last_updated": last_updated
        }

        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[content]
        )

    try:
        client.persist()
        logger.info("Database explicitly persisted.")
    except AttributeError:
        logger.info("Database automatically persisted on insert.")

    logger.info("Indexing process finished successfully.")


if __name__ == "__main__":
    index_protocols()
