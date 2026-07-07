"""
State Definition for the Medical Protocol Assistant RAG workflow.
"""
from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    """
    Represents the state of the RAG conversation/workflow.

    Attributes:
        user_query (str): The initial query typed by the user.
        medication (str): The identified medication name.
        department (str): The target department (e.g., ICU, Cardiology, General Ward).
        retrieved_documents (List[Dict[str, Any]]): List of protocols retrieved from ChromaDB.
        verification_result (Dict[str, Any]): Resolution status, conflict flags, and reasoning.
        answer (str): The final formulated safe answer/recommendation to display.
    """
    user_query: str
    medication: str
    department: str
    retrieved_documents: List[Dict[str, Any]]
    verification_result: Dict[str, Any]
    answer: str
