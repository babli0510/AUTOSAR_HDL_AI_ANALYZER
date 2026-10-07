from pathlib import Path
import sys


# ---------------------------------------------------------
# Project path setup
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT / "src"))

from vectorstore.vector_store import VectorStore


# ---------------------------------------------------------
# HLD Retriever
# ---------------------------------------------------------

class HLDRetriever:

    def __init__(self, top_k=3):

        self.top_k = top_k

        print("\nInitializing ChromaDB...")

        self.vector_store = VectorStore()

        print("Retriever initialized successfully.")


    # -----------------------------------------------------
    # Detect question type
    # -----------------------------------------------------

    def detect_query_type(self, query):

        query_lower = query.lower()

        # Dependency questions
        if any(word in query_lower for word in [
            "dependency",
            "depends on",
            "dependent",
            "relationship between"
        ]):
            return "dependency"


        # Signal questions
        if any(word in query_lower for word in [
            "signal",
            "signals"
        ]):
            return "signal"


        # Interface / port questions
        if any(word in query_lower for word in [
            "interface",
            "interfaces",
            "port",
            "ports"
        ]):
            return "interface"


        # Functional flow questions
        if any(word in query_lower for word in [
            "functional flow",
            "flow",
            "workflow",
            "sequence"
        ]):
            return "flow"


        # Component responsibility questions
        if any(word in query_lower for word in [
            "component",
            "responsibility",
            "responsibilities",
            "role"
        ]):
            return "component"


        # Architecture questions
        if any(word in query_lower for word in [
            "architecture",
            "architectural",
            "system overview"
        ]):
            return "architecture"


        return "general"


    # -----------------------------------------------------
    # Section relevance score
    # -----------------------------------------------------

    def calculate_section_score(
        self,
        query_type,
        section,
        content_type
    ):

        section_lower = section.lower()

        score = 0.0


        # -------------------------------------------------
        # Dependency
        # -------------------------------------------------

        if query_type == "dependency":

            if "dependency" in section_lower:
                score += 0.40

            if "architecture" in section_lower:
                score += 0.10


        # -------------------------------------------------
        # Signal
        # -------------------------------------------------

        elif query_type == "signal":

            if "signal" in section_lower:
                score += 0.45

            if "interface" in section_lower:
                score += 0.20


        # -------------------------------------------------
        # Interface
        # -------------------------------------------------

        elif query_type == "interface":

            if "interface" in section_lower:
                score += 0.45

            if "port" in section_lower:
                score += 0.30


        # -------------------------------------------------
        # Functional flow
        # -------------------------------------------------

        elif query_type == "flow":

            if "functional flow" in section_lower:
                score += 0.50

            if "dependency" in section_lower:
                score += 0.10


        # -------------------------------------------------
        # Component
        # -------------------------------------------------

        elif query_type == "component":

            if "architectural components" in section_lower:
                score += 0.45

            elif "architecture" in section_lower:
                score += 0.25


        # -------------------------------------------------
        # Architecture
        # -------------------------------------------------

        elif query_type == "architecture":

            if "architecture" in section_lower:
                score += 0.40

            if "system overview" in section_lower:
                score += 0.30


        # -------------------------------------------------
        # Structured content bonus
        # -------------------------------------------------

        if content_type == "structured":

            score += 0.10


        return score


    # -----------------------------------------------------
    # Retrieve relevant HLD chunks
    # -----------------------------------------------------

    def retrieve(self, query):

        # -------------------------------------------------
        # Get more candidates than final top_k
        # -------------------------------------------------

        candidate_count = max(
            self.top_k * 3,
            9
        )

        results = self.vector_store.search(
            query,
            candidate_count
        )


        if not results:
            return []


        # -------------------------------------------------
        # Detect question category
        # -------------------------------------------------

        query_type = self.detect_query_type(
            query
        )


        print(
            f"Query type detected: {query_type}"
        )


        # -------------------------------------------------
        # Re-rank results
        # -------------------------------------------------

        ranked_results = []


        for result in results:

            metadata = result.get(
                "metadata",
                {}
            )

            section = metadata.get(
                "section",
                ""
            )

            content_type = metadata.get(
                "content_type",
                "text"
            )

            distance = result.get(
                "distance"
            )


            # ------------------------------------------------
            # Semantic score
            #
            # Lower Chroma distance = better.
            # Convert it into a positive relevance score.
            # ------------------------------------------------

            if distance is None:

                semantic_score = 0.0

            else:

                semantic_score = 1.0 / (
                    1.0 + distance
                )


            # ------------------------------------------------
            # Section relevance
            # ------------------------------------------------

            section_score = (
                self.calculate_section_score(
                    query_type,
                    section,
                    content_type
                )
            )


            # ------------------------------------------------
            # Final ranking score
            # ------------------------------------------------

            final_score = (
                semantic_score * 0.70
                +
                section_score * 0.30
            )


            # Store scores for debugging
            result["_semantic_score"] = (
                semantic_score
            )

            result["_section_score"] = (
                section_score
            )

            result["_final_score"] = (
                final_score
            )

            ranked_results.append(
                result
            )


        # -------------------------------------------------
        # Sort highest score first
        # -------------------------------------------------

        ranked_results.sort(
            key=lambda x: x.get(
                "_final_score",
                0
            ),
            reverse=True
        )


        # -------------------------------------------------
        # Return only final top_k
        # -------------------------------------------------

        return ranked_results[
            :self.top_k
        ]


    # -----------------------------------------------------
    # Display retrieved results
    # -----------------------------------------------------

    def display_results(
        self,
        query,
        results
    ):

        print("\n" + "=" * 80)

        print(
            f"QUERY: {query}"
        )

        print("=" * 80)


        if not results:

            print(
                "\nNo relevant information found."
            )

            return


        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nRESULT {i}"
            )

            print(
                "-" * 80
            )


            # Chunk ID
            print(
                f"Chunk ID    : "
                f"{result['id']}"
            )


            # Metadata
            metadata = result.get(
                "metadata",
                {}
            )


            print(
                f"Page        : "
                f"{metadata.get('page_number', 'N/A')}"
            )


            print(
                f"Section     : "
                f"{metadata.get('section', 'N/A')}"
            )


            print(
                f"Content Type: "
                f"{metadata.get('content_type', 'N/A')}"
            )


            # Original Chroma distance
            distance = result.get(
                "distance"
            )


            if distance is not None:

                print(
                    f"Distance    : "
                    f"{distance:.4f}"
                )


            # Re-ranking scores
            print(
                f"Semantic    : "
                f"{result.get('_semantic_score', 0):.4f}"
            )


            print(
                f"Section     : "
                f"{result.get('_section_score', 0):.4f}"
            )


            print(
                f"Final Score : "
                f"{result.get('_final_score', 0):.4f}"
            )


            # Retrieved content
            print(
                "\nContent:"
            )


            document = result.get(
                "document",
                ""
            )

            print(document)


# ---------------------------------------------------------
# Test queries
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 80)

    print(
        "AUTOSAR HLD IMPROVED RETRIEVER TEST"
    )

    print("=" * 80)


    # -----------------------------------------------------
    # Create retriever
    # -----------------------------------------------------

    retriever = HLDRetriever(
        top_k=3
    )


    # -----------------------------------------------------
    # Test questions
    # -----------------------------------------------------

    test_queries = [

        "Which component provides VehicleState to ClimateController?",

        "What interfaces are used by ClimateController?",

        "What signals are provided by SensorManager?",

        "What is the responsibility of FanController?",

        "What is the dependency between VehicleStateManager and ClimateController?",

        "What is the functional flow of the Vehicle Climate Control ECU?"
    ]


    # -----------------------------------------------------
    # Run retrieval tests
    # -----------------------------------------------------

    for query in test_queries:

        results = retriever.retrieve(
            query
        )

        retriever.display_results(
            query,
            results
        )


    print(
        "\n" + "=" * 80
    )

    print(
        "IMPROVED RETRIEVER TEST COMPLETED"
    )

    print(
        "=" * 80
    )