import gradio as gr
import logging
from repochat1 import OpenAIEmbeddingRetriever, NVEmbedRetriever, load_datasets

# Define available datasets and models
datasets = ['movies', 'vibecheck', 'arena', 'math', 'narrativeqa', 'hotpot']
models = ['text-embedding-3-small', 'nvidia/NV-Embed-v2']

# Preload retrievers for each dataset and model
retrievers = {}
for dataset_name in datasets:
    print(f"Loading dataset: {dataset_name}")
    dataset = load_datasets([dataset_name])
    retrievers[dataset_name] = {}
    print(f"Building index for {dataset_name}...")
    for model_name in models:
        if model_name == "nvidia/NV-Embed-v2":
            retriever = NVEmbedRetriever(dataset)
            retriever.build_index()
            retrievers[dataset_name][model_name] = retriever
        else:
            retriever = OpenAIEmbeddingRetriever(dataset, model_name)
            retriever.build_index()
            retrievers[dataset_name][model_name] = retriever

def query_retriever(dataset_name, model_name, query, instruction):
    retriever = retrievers[dataset_name][model_name]
    retriever.instruction = instruction

    distances, indices = retriever.query(query)
    if distances is None or indices is None:
        return "Query failed."
    
    # Display the top retrieved documents along with their distances
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append(f"**Similarity: {dist:.4f} - Document:** {retriever.dataset[idx]}")
    return "\n\n--------\n\n".join(results)

# Create Gradio interface
iface = gr.Interface(
    fn=query_retriever,
    inputs=[
        gr.Dropdown(choices=datasets, label="Select Dataset"),
        gr.Dropdown(choices=models, label="Select Model"),
        gr.Textbox(label="Query"),
        gr.Textbox(label="Instruction", value="")
    ],
    outputs=gr.Markdown(),
    title="FAISS Retrieval System",
    description="Select a dataset and model, then input a query and optional instruction to retrieve documents. You can also flag interesting outputs which will be saved to a flagged folder."
)

if __name__ == "__main__":
    iface.launch(share=True) 