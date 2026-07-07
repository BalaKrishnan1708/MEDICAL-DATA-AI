"""
Librarian Agent Module.
Responsibility: retrieving medical protocols from ChromaDB without decision-making.
"""
import os
import time
import logging
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LibrarianAgent")


class LibrarianAgent:
    """
    Librarian Agent that connects to ChromaDB and retrieves protocols.
    """

    def __init__(self, db_path: str = None, model_name: str = "all-MiniLM-L6-v2"):
        if not db_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "chroma_db")

        logger.info("Librarian: Connecting to ChromaDB at %s", db_path)
        self.client = chromadb.PersistentClient(path=db_path)
        
        try:
            self.collection = self.client.get_collection("medical_protocols")
            logger.info("Librarian: Successfully loaded 'medical_protocols' collection.")
        except Exception as e:
            logger.error("Librarian: Failed to load collection. Ensure index_data.py has been run. Error: %s", str(e))
            self.collection = None

        logger.info("Librarian: Loading embedding model '%s'...", model_name)
        self.model = SentenceTransformer(model_name)

    def search_standard(self, medication: str) -> List[Dict[str, Any]]:
        if not self.collection:
            logger.warning("Librarian: Collection is not initialized.")
            return []

        start_time = time.perf_counter()
        query_text = f"Standard protocol for {medication}"
        logger.info("Librarian: Querying standard protocol semantic match for: %s", query_text)
        embedding = self.model.encode(query_text).tolist()

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=5,
            where={"medication": medication}
        )
        
        docs = self._format_results(results)
        standard_docs = [doc for doc in docs if doc["metadata"].get("protocol_type") == "Standard"]
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info("Librarian: Found %d standard protocol(s) [Time: %.1f ms]", len(standard_docs), elapsed)
        
        for doc in standard_docs:
            doc["search_metadata"] = {
                "elapsed_ms": elapsed,
                "similarity": doc.get("similarity", 0.0),
                "type": "Standard"
            }
        return standard_docs

    def search_override(self, medication: str) -> List[Dict[str, Any]]:
        if not self.collection:
            logger.warning("Librarian: Collection is not initialized.")
            return []

        start_time = time.perf_counter()
        query_text = f"Override protocol, ICU override, or safety alert for {medication}"
        logger.info("Librarian: Querying override/alert semantic match for: %s", query_text)
        embedding = self.model.encode(query_text).tolist()

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=5,
            where={"medication": medication}
        )
        
        docs = self._format_results(results)
        override_docs = [doc for doc in docs if doc["metadata"].get("protocol_type") == "Override"]
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info("Librarian: Found %d override protocol(s) [Time: %.1f ms]", len(override_docs), elapsed)
        
        for doc in override_docs:
            doc["search_metadata"] = {
                "elapsed_ms": elapsed,
                "similarity": doc.get("similarity", 0.0),
                "type": "Override"
            }
        return override_docs

    def multi_hop_search(self, medication: str) -> List[Dict[str, Any]]:
        logger.info("Librarian: Initiating multi-hop search for medication: %s", medication)
        standard_docs = self.search_standard(medication)
        override_docs = self.search_override(medication)

        seen_ids = set()
        combined_docs = []
        for doc in standard_docs + override_docs:
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                combined_docs.append(doc)

        logger.info("Librarian: Multi-hop search returned %d unique document(s) in total", len(combined_docs))
        return combined_docs

    def metadata_filter_search(self, medication: str) -> List[Dict[str, Any]]:
        if not self.collection:
            logger.warning("Librarian: Collection is not initialized.")
            return []

        start_time = time.perf_counter()
        logger.info("Librarian: Starting strict metadata-filtered search for medication: %s", medication)
        results = self.collection.get(
            where={"medication": medication}
        )

        docs = []
        if results and results.get("ids"):
            for i in range(len(results["ids"])):
                docs.append({
                    "id": results["ids"][i],
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "similarity": 1.0
                })

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info("Librarian: Metadata filter search retrieved %d document(s) [Time: %.1f ms]", len(docs), elapsed)
        
        for doc in docs:
            doc["search_metadata"] = {
                "elapsed_ms": elapsed,
                "similarity": 1.0,
                "type": "Metadata-Filtered"
            }
        return docs

    def retrieve(self, state: Dict[str, Any]) -> Dict[str, Any]:
        medication = state.get("medication", "")
        docs = self.search_standard(medication)
        return {"retrieved_documents": docs}

    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        formatted = []
        if not results or "ids" not in results or not results["ids"]:
            return formatted

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else None

        for i, doc_id in enumerate(ids):
            distance_val = distances[i] if distances and i < len(distances) else 0.0
            similarity = max(0.0, min(1.0, 1.0 - distance_val))
            formatted.append({
                "id": doc_id,
                "content": documents[i],
                "metadata": metadatas[i],
                "similarity": similarity
            })
        return formatted
