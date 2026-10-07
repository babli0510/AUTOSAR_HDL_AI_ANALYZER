from pathlib import Path
import sys
import requests


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT / "src"))


# ============================================================
# IMPORT RETRIEVER
# ============================================================

from rag.retriever import HLDRetriever


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Smaller model because of available system RAM
MODEL_NAME = "qwen2.5:0.5b"


# ============================================================
# RAG ANSWER GENERATOR
# ============================================================

class RAGAnswerGenerator:

    def __init__(self, top_k=3):

        print("\nInitializing RAG Answer Generator...")

        self.retriever = HLDRetriever(top_k=top_k)

        self.model_name = MODEL_NAME

        print("RAG Answer Generator initialized successfully.")


    # ========================================================
    # BUILD GROUNDED PROMPT
    # ========================================================

    def build_prompt(self, question, results):

        if not results:
            return None

        context_parts = []

        for i, result in enumerate(results, start=1):

            metadata = result.get("metadata", {})

            page = metadata.get(
                "page_number",
                "Unknown"
            )

            section = metadata.get(
                "section",
                "Unknown"
            )

            content = result.get(
                "document",
                ""
            )

            context_parts.append(
                f"""
SOURCE {i}
Page: {page}
Section: {section}

{content}
"""
            )

        context = "\n".join(context_parts)

        # ====================================================
        # STRICT GROUNDED RAG PROMPT
        # ====================================================

        prompt = f"""
You are an AUTOSAR High-Level Design (HLD)
document analysis assistant.

Answer the user's question ONLY using the
provided HLD evidence.

STRICT RULES:

1. Use only information present in the evidence.

2. Do not use outside knowledge.

3. Do not invent components, interfaces,
   signals, dependencies, relationships,
   responsibilities, or values.

4. If the evidence does not contain enough
   information, say:

"The information is not available in
the provided HLD evidence."

5. Keep the answer concise and technical.

6. Mention the supporting page and section.

7. Do not provide unsupported assumptions.

USER QUESTION:

{question}

HLD EVIDENCE:

{context}

Provide the grounded answer.
"""

        return prompt


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    def generate_answer(self, question):

        print("\nRetrieving relevant HLD evidence...")

        # ----------------------------------------------------
        # RETRIEVE
        # ----------------------------------------------------

        results = self.retriever.retrieve(question)

        if not results:

            return {
                "answer": (
                    "The information is not available "
                    "in the provided HLD evidence."
                ),
                "sources": []
            }

        print(
            f"Retrieved {len(results)} relevant HLD chunks."
        )

        # ----------------------------------------------------
        # BUILD PROMPT
        # ----------------------------------------------------

        prompt = self.build_prompt(
            question,
            results
        )

        print(
            f"Prompt length: {len(prompt)} characters"
        )

        # ----------------------------------------------------
        # SEND TO OLLAMA
        # ----------------------------------------------------

        print(
            f"Sending evidence to {self.model_name}..."
        )

        payload = {
            "model": self.model_name,

            "prompt": prompt,

            "stream": False,

            "options": {
                # Deterministic answer
                "temperature": 0,

                # Very small context to reduce RAM usage
                "num_ctx": 512
            }
        }

        try:

            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=180
            )

            print(
                f"Ollama HTTP Status: "
                f"{response.status_code}"
            )

            response.raise_for_status()

            data = response.json()

            answer = data.get(
                "response",
                ""
            ).strip()

            # ------------------------------------------------
            # COLLECT SOURCES
            # ------------------------------------------------

            sources = []

            for result in results:

                metadata = result.get(
                    "metadata",
                    {}
                )

                sources.append({
                    "page": metadata.get(
                        "page_number"
                    ),

                    "section": metadata.get(
                        "section"
                    ),

                    "chunk_id": result.get(
                        "id"
                    ),

                    "distance": result.get(
                        "distance"
                    )
                })

            return {
                "answer": answer,
                "sources": sources
            }

        # ----------------------------------------------------
        # HTTP ERROR
        # ----------------------------------------------------

        except requests.exceptions.HTTPError as error:

            print("\nOLLAMA HTTP ERROR")
            print("=" * 80)

            print(
                f"Status Code: "
                f"{response.status_code}"
            )

            print("\nOllama Response:")

            print(response.text)

            print("=" * 80)

            raise error

        # ----------------------------------------------------
        # CONNECTION ERROR
        # ----------------------------------------------------

        except requests.exceptions.ConnectionError:

            print("\nOLLAMA CONNECTION ERROR")
            print("=" * 80)

            print(
                "Could not connect to Ollama."
            )

            print(
                "Make sure 'ollama serve' is "
                "running in another CMD."
            )

            print("=" * 80)

            raise


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    def display_answer(
        self,
        question,
        result
    ):

        print("\n")

        print("=" * 80)
        print("QUESTION")
        print("=" * 80)

        print(question)

        print("\n")

        print("=" * 80)
        print("GROUNDED ANSWER")
        print("=" * 80)

        print(
            result["answer"]
        )

        print("\n")

        print("=" * 80)
        print("SOURCES")
        print("=" * 80)

        if not result["sources"]:

            print("No sources available.")

        else:

            for source in result["sources"]:

                distance = source.get(
                    "distance"
                )

                if distance is not None:

                    print(
                        f"Page {source['page']} | "
                        f"Section: {source['section']} | "
                        f"Chunk: {source['chunk_id']} | "
                        f"Distance: {distance:.4f}"
                    )

                else:

                    print(
                        f"Page {source['page']} | "
                        f"Section: {source['section']} | "
                        f"Chunk: {source['chunk_id']}"
                    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 80)

    print(
        "AUTOSAR HLD RAG ANSWER GENERATOR"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # INITIALIZE RAG
    # --------------------------------------------------------

    rag = RAGAnswerGenerator(
        top_k=3
    )

    # --------------------------------------------------------
    # TEST QUESTION
    # --------------------------------------------------------

    question = (
        "Which component provides VehicleState "
        "to ClimateController?"
    )

    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    try:

        result = rag.generate_answer(
            question
        )

        rag.display_answer(
            question,
            result
        )

    except requests.exceptions.ConnectionError:

        print("\nERROR:")

        print(
            "Could not connect to Ollama."
        )

        print(
            "Make sure 'ollama serve' is "
            "running in another CMD."
        )

    except requests.exceptions.HTTPError:

        print("\nERROR:")

        print(
            "Ollama returned an HTTP error."
        )

        print(
            "Check the Ollama response shown above."
        )

    except Exception as error:

        print("\nERROR:")

        print(
            str(error)
        )