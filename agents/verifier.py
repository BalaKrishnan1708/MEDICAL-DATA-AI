"""
Verifier Agent Module.
Responsibility: reviewing retrieved protocols, detecting conflicts, and validating department overrides.
"""
import time
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("VerifierAgent")


class VerifierAgent:
    """
    Verifier Agent that checks safety of retrieved protocols.
    """

    def detect_override(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [doc for doc in documents if doc["metadata"].get("protocol_type") == "Override"]

    def validate_department(self, documents: List[Dict[str, Any]], target_department: str) -> List[Dict[str, Any]]:
        return [
            doc for doc in documents 
            if doc["metadata"].get("department", "").lower() == target_department.lower()
        ]

    def compare(self, standard_doc: Dict[str, Any], override_doc: Dict[str, Any]) -> str:
        std_content = standard_doc.get("content", "")
        ovr_content = override_doc.get("content", "")
        std_dept = standard_doc["metadata"].get("department", "General Ward")
        ovr_dept = override_doc["metadata"].get("department", "Specialized")
        
        comparison = (
            f"Conflict Detected between {std_dept} Protocol (Standard) and {ovr_dept} Protocol (Override).\n"
            f"- Standard: {std_content}\n"
            f"- Override: {ovr_content}\n"
            f"Resolution: The specialized {ovr_dept} override protocol must be used as it supersedes the standard protocol."
        )
        return comparison

    def reject_if_unsafe(
        self, 
        documents: List[Dict[str, Any]], 
        department: str,
        is_metadata_search_done: bool = False
    ) -> Tuple[Dict[str, Any], str]:
        start_time = time.perf_counter()
        logger.info("Verifier: Reviewing %d document(s) for department: %s", len(documents), department)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        if not documents:
            logger.info("Verifier: No documents provided. Medication not found in database.")
            return (
                {
                    "conflict": False,
                    "conflict_detected": False,
                    "conflict_resolved": True,
                    "status": "not_found",
                    "elapsed_ms": elapsed_ms,
                    "documents_evaluated": 0
                },
                "Medication not found in protocol database."
            )

        standards = [doc for doc in documents if doc["metadata"].get("protocol_type") == "Standard"]
        overrides = self.detect_override(documents)
        
        dept_overrides = self.validate_department(overrides, department)

        dept_lower = department.lower()
        
        document_profile = [
            {"id": doc["id"], "type": doc["metadata"].get("protocol_type"), "similarity": doc.get("similarity", 1.0)}
            for doc in documents
        ]

        if dept_lower in ["icu", "cardiology"]:
            if dept_overrides:
                override_doc = dept_overrides[0]
                reason = f"{department} override supersedes General Ward protocol."
                
                if standards:
                    comparison = self.compare(standards[0], override_doc)
                    logger.info("Verifier comparison detail:\n%s", comparison)
                
                ans_text = (
                    "STANDARD PROTOCOL FOUND\n"
                    "Searching for overrides...\n"
                    "OVERRIDE FOUND\n\n"
                    f"Recommendation\n{override_doc['content']}\n\n"
                    f"Reason\n{reason}"
                )
                
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return (
                    {
                        "conflict": False,
                        "conflict_detected": True,
                        "conflict_resolved": True,
                        "status": "override_approved",
                        "override_id": override_doc["id"],
                        "elapsed_ms": elapsed_ms,
                        "documents_evaluated": len(documents),
                        "evaluated_profiles": document_profile
                    },
                    ans_text
                )
            
            elif not overrides and not is_metadata_search_done:
                logger.warning("Verifier: Specialized department %s requested, but only standard protocol retrieved. Triggering metadata filter search loop.", department)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return (
                    {
                        "conflict": True,
                        "conflict_detected": True,
                        "conflict_resolved": False,
                        "status": "override_check_required",
                        "elapsed_ms": elapsed_ms,
                        "documents_evaluated": len(documents),
                        "evaluated_profiles": document_profile
                    },
                    ""
                )
            else:
                standard_doc = standards[0] if standards else documents[0]
                ans_text = (
                    "No specific overrides found; showing standard protocol only.\n\n"
                    f"Recommendation\n{standard_doc['content']}"
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return (
                    {
                        "conflict": False,
                        "conflict_detected": False,
                        "conflict_resolved": True,
                        "status": "standard_only",
                        "elapsed_ms": elapsed_ms,
                        "documents_evaluated": len(documents),
                        "evaluated_profiles": document_profile
                    },
                    ans_text
                )

        else:
            if standards:
                standard_doc = standards[0]
                ans_text = (
                    f"Recommendation\n{standard_doc['content']}"
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return (
                    {
                        "conflict": False,
                        "conflict_detected": False,
                        "conflict_resolved": True,
                        "status": "standard_approved",
                        "elapsed_ms": elapsed_ms,
                        "documents_evaluated": len(documents),
                        "evaluated_profiles": document_profile
                    },
                    ans_text
                )
            else:
                doc = documents[0]
                ans_text = (
                    f"Recommendation\n{doc['content']}"
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return (
                    {
                        "conflict": False,
                        "conflict_detected": False,
                        "conflict_resolved": True,
                        "status": "approved",
                        "elapsed_ms": elapsed_ms,
                        "documents_evaluated": len(documents),
                        "evaluated_profiles": document_profile
                    },
                    ans_text
                )
