"""
CLI Interface Application.
Prompts user for Medication and Department, runs the LangGraph workflow, and prints the result.
"""
import sys
import logging
from scripts.graph import build_graph

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("App")

logging.getLogger("GraphManager").setLevel(logging.INFO)
logging.getLogger("LibrarianAgent").setLevel(logging.INFO)
logging.getLogger("VerifierAgent").setLevel(logging.INFO)


def main():
    """
    Main application loop. Runs the medical protocol assistant.
    """
    print("=" * 60)
    print("      MEDICAL PROTOCOL ASSISTANT (MULTI-AGENT RAG)      ")
    print("=" * 60)
    print("Initializing agents and loading database, please wait...")

    try:
        graph = build_graph()
    except Exception as e:
        print(f"\nError: Could not initialize LangGraph or database. {e}")
        print("Please verify that ChromaDB exists and scripts/index_data.py has been run successfully.")
        sys.exit(1)

    print("\nSystem ready! (Press Ctrl+C to exit)\n")

    while True:
        try:
            print("-" * 50)
            medication = input("Medication:\n").strip()
            if not medication:
                print("Medication name cannot be empty. Please try again.")
                continue

            department = input("\nDepartment:\n").strip()
            if not department:
                print("Department name cannot be empty. Please try again.")
                continue

            print("\nProcessing request through Librarian and Safety Verifier Agents...")
            
            initial_state = {
                "user_query": f"Retrieve protocol for {medication} in {department}",
                "medication": medication,
                "department": department,
                "retrieved_documents": [],
                "verification_result": {},
                "answer": ""
            }

            final_output = graph.invoke(initial_state)
            vr = final_output.get("verification_result", {})
            
            print("\nOutput")
            print("---------------------")
            print(final_output.get("answer", "No answer generated."))
            print("---------------------")
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
            print("-" * 50)
            print("\n")

        except KeyboardInterrupt:
            print("\nExiting Medical Protocol Assistant. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred while processing: {e}")
            logger.exception("Details:")
            print("\n")


if __name__ == "__main__":
    main()
