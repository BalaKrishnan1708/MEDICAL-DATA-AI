"""
LangGraph Workflow Module.
Defines the cyclic StateGraph structure containing Librarian, Verifier, and Routing edges.
"""
import os
import sys
import logging
import time
from langgraph.graph import StateGraph, START, END

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from agents.state import AgentState
from agents.librarian import LibrarianAgent
from agents.verifier import VerifierAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GraphManager")


def build_graph():
    """
    Build and compile the LangGraph workflow state machine.
    """
    logger.info("Graph: Initializing agents and building workflow...")
    
    librarian = LibrarianAgent()
    verifier = VerifierAgent()

    workflow = StateGraph(AgentState)

    def librarian_node(state: AgentState) -> dict:
        start_time = time.perf_counter()
        logger.info("========================================")
        logger.info("[NODE: Librarian Retrieve]")
        logger.info("Retrieving standard protocol for medication: %s", state.get("medication"))
        res = librarian.retrieve(state)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Librarian Retrieve completed in %.2f ms", elapsed_ms)
        return res

    def verifier_node(state: AgentState) -> dict:
        start_time = time.perf_counter()
        logger.info("========================================")
        logger.info("[NODE: Verifier]")
        
        prev_result = state.get("verification_result", {})
        is_metadata_done = prev_result.get("metadata_search_done", False)

        result, answer = verifier.reject_if_unsafe(
            documents=state.get("retrieved_documents", []),
            department=state.get("department", ""),
            is_metadata_search_done=is_metadata_done
        )

        if prev_result.get("metadata_search_done"):
            result["metadata_search_done"] = True

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        result["node_elapsed_ms"] = elapsed_ms
        
        logger.info("Verification Node completed in %.2f ms", elapsed_ms)
        logger.info("Decision Metadata: conflict_detected=%s, conflict_resolved=%s, loop_back=%s, status=%s",
                    result.get("conflict_detected"), result.get("conflict_resolved"), result.get("conflict"), result.get("status"))
        
        return {
            "verification_result": result,
            "answer": answer
        }

    def librarian_metadata_search_node(state: AgentState) -> dict:
        start_time = time.perf_counter()
        logger.info("========================================")
        logger.info("[NODE: Librarian Metadata Search]")
        medication = state.get("medication", "")
        logger.info("Performing metadata-filtered search for medication: %s", medication)
        
        docs = librarian.metadata_filter_search(medication)
        
        result = state.get("verification_result", {})
        result["metadata_search_done"] = True
        result["conflict"] = False

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Librarian Metadata Search Node completed in %.2f ms", elapsed_ms)
        return {
            "retrieved_documents": docs,
            "verification_result": result
        }

    workflow.add_node("librarian", librarian_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("librarian_metadata_search", librarian_metadata_search_node)

    workflow.add_edge(START, "librarian")
    workflow.add_edge("librarian", "verifier")

    def router(state: AgentState) -> str:
        res = state.get("verification_result", {})
        if res.get("conflict"):
            logger.info("Verifier requested override lookup. Routing to [Librarian Metadata Search]")
            return "librarian_metadata_search"
        else:
            logger.info("Routing directly to [END]")
            return END

    workflow.add_conditional_edges(
        "verifier",
        router,
        {
            "librarian_metadata_search": "librarian_metadata_search",
            END: END
        }
    )

    workflow.add_edge("librarian_metadata_search", "verifier")

    app = workflow.compile()
    logger.info("Graph compiled successfully.")
    return app
