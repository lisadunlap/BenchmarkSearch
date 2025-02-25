#!/usr/bin/env python3
import argparse
import logging
from abc import ABC, abstractmethod

import faiss
import numpy as np
from tqdm import tqdm

# Import the LLM embedding function from your llm_utils file
from serve.utils_llm import get_llm_embedding

# Import your dataset loader
from data_prep import load_datasets

logging.basicConfig(level=logging.INFO)


class BaseRetrieval(ABC):
    """
    Abstract base retrieval class. It handles building a FAISS index
    for a given dataset by computing embeddings and provides a query method.
    """

    def __init__(self, dataset, instruction=""):
        self.dataset = dataset
        self.instruction = instruction
        self.embeddings = None
        self.index = None

    @abstractmethod
    def get_embedding(self, text):
        """
        Abstract method to compute the embedding for a given text.
        Must be implemented by child classes.
        """
        pass

    def build_index(self):
        """
        Compute embeddings for each document in the dataset and build a FAISS index.
        """
        logging.info("Computing embeddings for dataset...")
        embeddings_list = []
        # Wrap the dataset with tqdm to show a progress bar
        for doc in tqdm(self.dataset, desc="Computing embeddings"):
            emb = self.get_embedding(doc)
            if emb is None:
                logging.warning("Embedding returned None for a document; skipping.")
                continue
            embeddings_list.append(emb)
        if not embeddings_list:
            logging.error("No embeddings were computed. Exiting.")
            return
        # Convert list to numpy array (assumes each embedding is a 1-D vector)
        self.embeddings = np.vstack(embeddings_list).astype("float32")
        # Normalize the embeddings
        faiss.normalize_L2(self.embeddings)
        dimension = self.embeddings.shape[1]
        logging.info(
            f"Building FAISS index with dimension {dimension} for {self.embeddings.shape[0]} documents"
        )
        # Use IndexFlatIP for cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)
        logging.info("FAISS index built successfully.")

    def query(self, query_str, k=5):
        """
        Compute the query embedding, perform a search on the FAISS index,
        and return the top k results.
        """
        logging.info(f"Computing embedding for query: {query_str}")
        query_emb = self.get_embedding(query_str)
        if query_emb is None:
            logging.error("Failed to compute query embedding.")
            return None, None
        query_emb = np.array(query_emb).astype("float32").reshape(1, -1)
        # Normalize the query embedding
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, k)
        return distances, indices


class OpenAIEmbeddingRetriever(BaseRetrieval):
    """
    Retrieval class for OpenAI-based embedding models.
    """

    def __init__(self, dataset, model_name, instruction=""):
        super().__init__(dataset, instruction)
        self.model_name = model_name

    def get_embedding(self, text):
        return get_llm_embedding(text, self.model_name, self.instruction)


class NVEmbedRetriever(BaseRetrieval):
    """
    Retrieval class for the NVIDIA NV-Embed-v2 model.
    """

    def __init__(self, dataset, instruction=""):
        super().__init__(dataset, instruction)
        self.model_name = "nvidia/NV-Embed-v2"

    def get_embedding(self, text):
        return get_llm_embedding(text, self.model_name, self.instruction)


def run_test():
    """
    Quick test function using a dummy dataset.
    """
    print("Running quick test...")
    # Define a small dummy dataset
    test_dataset = [
        "The cat sat on the mat.",
        "The quick brown fox jumped over the lazy dog.",
        "OpenAI is a research laboratory focused on artificial intelligence.",
        "FAISS is a library for efficient similarity search."
    ]
    test_model = "text-embedding-3-small"  # Change this if you wish to test a different model
    test_instruction = ""
    test_query = "Tell me something about cats."
    retriever = OpenAIEmbeddingRetriever(test_dataset, test_model, test_instruction)
    retriever.build_index()
    distances, indices = retriever.query(test_query, k=3)
    print("Test Query Results:")
    for dist, idx in zip(distances[0], indices[0]):
        print(f"Distance: {dist:.4f} - Document: {test_dataset[idx]}")
    assert indices[0][0] == 0


def main():
    parser = argparse.ArgumentParser(
        description="Retrieval system using FAISS index and LLM embeddings."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick test with a dummy dataset and exit.",
    )
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument(
        "--embedding_model",
        type=str,
        help="Embedding model name (e.g., 'text-embedding-3-small', 'nvidia/NV-Embed-v2')",
    )
    parser.add_argument("--query", type=str, help="Query string")
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Optional instruction for the LLM embedding",
    )
    args = parser.parse_args()

    # If test flag is set, run the test and exit
    if args.test:
        run_test()
        return
    
    print(f"{args.dataset=}")
    print(f"{args.embedding_model=}")
    print(f"{args.query=}")
    print(f"{args.instruction=}")

    # Ensure all required arguments are provided
    if not (args.dataset and args.embedding_model and args.query):
        parser.error("dataset, embedding_model, and query are required unless --test is used.")

    # Load dataset (assumed to return a list of documents)
    logging.info(f"Loading dataset: {args.dataset}")
    dataset = load_datasets([args.dataset])
    print(f"{dataset=}")
    if not dataset:
        logging.error("Dataset is empty or not found. Exiting.")
        return

    # Instantiate the appropriate retriever based on the embedding model
    if args.embedding_model == "nvidia/NV-Embed-v2":
        print("Using NVIDIA NV-Embed-v2")
        retriever = NVEmbedRetriever(dataset, args.instruction)
    else:
        print(f"Using {args.embedding_model}")
        retriever = OpenAIEmbeddingRetriever(dataset, args.embedding_model, args.instruction)

    # Build the FAISS index
    retriever.build_index()
    if retriever.index is None:
        logging.error("Index building failed. Exiting.")
        return

    # Perform query
    distances, indices = retriever.query(args.query)
    if distances is None or indices is None:
        logging.error("Query failed. Exiting.")
        return

    # Display the top retrieved documents along with their distances
    logging.info("Top retrieved documents:")
    for dist, idx in zip(distances[0], indices[0]):
        logging.info(f"Distance: {dist:.4f} - Document: {dataset[idx]}")


if __name__ == "__main__":
    main()