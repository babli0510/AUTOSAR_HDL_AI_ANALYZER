from pathlib import Path
import sys

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vectorstore"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "autosar_hld_chunks"


# ---------------------------------------------------------
# Vector Store
# ---------------------------------------------------------

class VectorStore:

    def __init__(self):

        print("\nInitializing ChromaDB...")

        # Load embedding model
        self.embedding_model = SentenceTransformer(
            MODEL_NAME
        )

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_PATH)
        )

        # Create or load collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        print("ChromaDB initialized successfully.")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Database path: {VECTOR_DB_PATH}")


    # -----------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------

    def generate_embeddings(self, texts):

        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings.tolist()


    # -----------------------------------------------------
    # Add documents to vector database
    # -----------------------------------------------------

    def add_documents(self, chunks):

        if not chunks:
            print("No chunks provided.")
            return

        documents = []
        ids = []
        metadatas = []

        for chunk in chunks:

            documents.append(
                chunk["text"]
            )

            ids.append(
                chunk["chunk_id"]
            )

            metadata = {
                "page_number": int(
                    chunk["page_number"]
                ),
                "section": str(
                    chunk["section"]
                ),
                "chunk_number": int(
                    chunk["chunk_number"]
                ),
                "content_type": str(
                    chunk["content_type"]
                )
            }

            metadatas.append(metadata)

        # Generate embeddings
        embeddings = self.generate_embeddings(
            documents
        )

        # Store in ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        print(
            f"Stored {len(documents)} chunks in ChromaDB."
        )


    # -----------------------------------------------------
    # Search relevant documents
    # -----------------------------------------------------

    def search(self, query, top_k=3):

        # -------------------------------------------------
        # Convert question into embedding
        # -------------------------------------------------

        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        )

        query_embedding = query_embedding.tolist()

        # -------------------------------------------------
        # Search ChromaDB
        # -------------------------------------------------

        results = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k
        )

        # -------------------------------------------------
        # Format results
        # -------------------------------------------------

        formatted_results = []

        ids = results.get("ids", [[]])[0]

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        for i in range(len(ids)):

            formatted_results.append({

                "id": ids[i],

                "document": documents[i],

                "metadata": metadatas[i],

                "distance": distances[i]
                if i < len(distances)
                else None
            })

        return formatted_results


    # -----------------------------------------------------
    # Count stored documents
    # -----------------------------------------------------

    def count(self):

        return self.collection.count()


# ---------------------------------------------------------
# Test Vector Store
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 80)
    print("AUTOSAR HLD VECTOR DATABASE TEST")
    print("=" * 80)

    vector_store = VectorStore()

    print(
        f"\nDocuments currently stored: "
        f"{vector_store.count()}"
    )

    # Test semantic search
    test_query = (
        "Which component provides VehicleState "
        "to ClimateController?"
    )

    print("\nTest Query:")
    print(test_query)

    results = vector_store.search(
        test_query,
        top_k=3
    )

    print("\nSearch Results:")

    for i, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 80)

        print(f"Result {i}")
        print(f"Chunk ID: {result['id']}")

        print(
            f"Page: "
            f"{result['metadata']['page_number']}"
        )

        print(
            f"Section: "
            f"{result['metadata']['section']}"
        )

        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )

        print("\nContent:")
        print(result["document"])

    print("\n" + "=" * 80)
    print("VECTOR DATABASE TEST COMPLETED")
    print("=" * 80)