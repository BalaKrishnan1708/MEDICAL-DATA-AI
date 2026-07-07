"""
Automated Test Case Runner.
Executes the LangGraph pipeline for the six required test cases and prints validation outputs.
"""
import os
import sys
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("GraphManager").setLevel(logging.INFO)
logging.getLogger("LibrarianAgent").setLevel(logging.INFO)
logging.getLogger("VerifierAgent").setLevel(logging.INFO)

from scripts.graph import build_graph


def run_test_case(graph, medication: str, department: str):
    """
    Run a single test case through the compiled graph.
    """
    print("\n" + "=" * 70)
    print(f" TEST RUN: Medication = '{medication}' | Department = '{department}'")
    print("=" * 70)
    
    initial_state = {
        "user_query": f"Retrieve protocol for {medication} in {department}",
        "medication": medication,
        "department": department,
        "retrieved_documents": [],
        "verification_result": {},
        "answer": ""
    }
    
    final_state = graph.invoke(initial_state)
    vr = final_state.get("verification_result", {})
    
    print("\n----------------- OUTPUT -----------------")
    print(final_state.get("answer"))
    print("------------------------------------------")
    print("Safety Verification Analytics:")
    print(f"- Status: {vr.get('status')}")
    print(f"- Conflict Detected: {vr.get('conflict_detected')}")
    print(f"- Conflict Resolved: {vr.get('conflict_resolved')}")
    print(f"- Total Documents Evaluated: {vr.get('documents_evaluated', 0)}")
    if vr.get('evaluated_profiles'):
        print("  Retrieved Protocols Profile:")
        for prof in vr['evaluated_profiles']:
            print(f"  * ID: {prof['id']} | Type: {prof['type']} | Similarity: {prof['similarity']:.4f}")
    print(f"- Verification Node Latency: {vr.get('node_elapsed_ms', 0.0):.2f} ms")
    print("=" * 70 + "\n")


def main():
    """
    Build graph and execute all 6 test cases sequentially.
    """
    print("Building LangGraph workflow...")
    graph = build_graph()
    
    test_cases = [
        ("Insulin Lispro", "ICU"),
        ("Insulin Lispro", "General Ward"),
        ("Heparin", "Cardiology"),
        ("Potassium Chloride", "ICU"),
        ("Potassium Chloride", "General Ward"),
        ("Aspirin", "ICU")
    ]
    
    for med, dept in test_cases:
        run_test_case(graph, med, dept)


if __name__ == "__main__":
    main()
