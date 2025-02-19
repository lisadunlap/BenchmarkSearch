import threading
import faiss
import numpy as np
import gradio as gr
import lmdb
from litellm import embedding
from data_prep import load_datasets  # Assuming data_prep.py is in the same directory
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils_llm import get_llm_embedding

def setup_oai_retrieval():
    # Load datasets
    # documents = load_datasets(['arena', 'math', 'narrativeqa', 'movies', 'vibecheck'])
    documents = load_datasets(['vibecheck'])
    # documents = list(set(documents))
    # Example usage with 10 workers
    print("Computing embeddings...")
    document_embeddings = get_llm_embedding(list(documents), model="text-embedding-3-small")
    print("Embeddings computed.")

    # Convert embeddings to numpy array
    document_embeddings = np.array(document_embeddings).astype('float32')

    # Build FAISS index
    dimension = document_embeddings.shape[1]
    # Use IndexFlatIP for direct cosine similarity computation
    index = faiss.IndexFlatIP(dimension)

    # Normalize vectors (L2 norm) - this is still needed
    faiss.normalize_L2(document_embeddings)
    index.add(document_embeddings)
    print(f"FAISS index built with {index.ntotal} documents.")
    return documents, index

# Define the retrieval function
def retrieve(query, k, index, documents):
    # Get query embedding and reshape it properly
    query_embedding = get_llm_embedding(query, model="text-embedding-3-small")
    query_embedding = np.array([query_embedding]).astype('float32')
    
    # Normalize query vector
    faiss.normalize_L2(query_embedding)
    
    # Using IndexFlatIP, the distances are already cosine similarities
    similarities, indices = index.search(query_embedding, k)
    
    # Format results (note: no need to convert distances anymore)
    results = []
    for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
        if idx != -1:  # Check for valid index
            results.append(f"**__Rank {i+1}.__** (Similarity: {similarity:.3f}) {documents[idx]}")
    
    return "\n\n".join(results) if results else "No matching documents found."

# Define the retrieval function
def retrieve_fn(query: str, example: str, instruction: str, top_k: int) -> str:
    """
    Retrieve results using optional custom instruction for the main query.
    """
    try:
        top_k_val = int(top_k)
    except ValueError:
        top_k_val = 3

    output_lines = []

    # Standard retrieval using "prompt description" if available
    if len(instruction) > 0:
        query = instruction + ": " + query
    else:
        print("No instruction provided")
    if query.strip():
        results = retrieve(query, top_k_val)
        if results:
            output_lines.append("## Top-k for Prompt Description Retrieval")
            output_lines.append(results)
        else:
            output_lines.append("No results returned for your main prompt description query.")

    if not output_lines:
        return "No query provided. Please enter at least one."
    else:
        return "\n".join(output_lines)

# Set up Gradio interface
def build_gradio_app():
    """
    Build and return a Gradio Blocks interface for the retrieval system.
    """
    with gr.Blocks() as demo:
        gr.Markdown("# A Cute Little Prompt Retrieval System (with optional Q2Q)")        
        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="Enter your query/prompt description:",
                    placeholder="e.g. 'creative writing questions for generating long nonfiction stories'"
                )
                example_input = gr.Textbox(
                    label="Example problem (optional)",
                    placeholder="e.g. 'a combinatorics question: There are 10 people...' (for Q2Q retrieval)"
                )
                instruction_input = gr.Textbox(
                    label="Custom instruction (optional)",
                    placeholder="e.g. 'Given the description of a type of question, retrieve passages that contain questions of that type'"
                )
                top_k_input = gr.Number(
                    label="Top K", 
                    value=5, 
                    precision=0
                )
                submit_btn = gr.Button("Retrieve")
            with gr.Column(scale=2):
                output_box = gr.Markdown(label="Retrieved Passages")

        submit_btn.click(
            fn=retrieve_fn, 
            inputs=[query_input, example_input, instruction_input, top_k_input], 
            outputs=[output_box]
        )

    return demo

if __name__ == "__main__":
    documents, index = setup_oai_retrieval()
    demo = build_gradio_app()
    demo.launch(share=True)