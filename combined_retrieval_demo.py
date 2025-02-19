import gradio as gr
from toy_retrival_system import RetrievalSystem as NVRetrievalSystem
from oai_retrival_system import retrieve as oai_retrieve
from oai_retrival_system import setup_oai_retrieval
from data_prep import load_datasets

def build_combined_demo():
    # Initialize NV-Embed retriever
    dataset_names = ['arena', 'math', 'narrativeqa', 'movies', 'vibecheck']
    passages = load_datasets(dataset_names)
    print(f"Loaded {len(passages)} passages")
    nv_retriever = NVRetrievalSystem(
        model_name='nvidia/NV-Embed-v2',
        lmdb_path="embedding_cache.lmdb",
        map_size=2**32,
        use_faiss=True,
        device='cuda',
        num_gpus=2
    )
    
    # Build FAISS index for NV-Embed
    nv_retriever.build_faiss_index(passages, batch_size=32)

    documents, index = setup_oai_retrieval()
    
    # Define retrieval functions for each system
    def nv_retrieve_fn(query: str, example: str, instruction: str, top_k: int) -> str:
        try:
            top_k_val = int(top_k)
        except ValueError:
            top_k_val = 3

        output_lines = []

        if query.strip():
            results = nv_retriever.retrieve([query], top_k=top_k_val, instruction=instruction)
            if results:
                output_lines.append("## Top-k for Prompt Description Retrieval")
                retrieved = results[0]["retrieved_passages"]
                for rank, (passage, score) in enumerate(retrieved, start=1):
                    output_lines.append(f"**Rank {rank}** | Score: {score:.4f}")
                    # Check if the passage contains backticks or code-like content
                    if '`' in passage or any(char in passage for char in ['{', '}', ';']):
                        output_lines.append("```")
                        output_lines.append(passage)
                        output_lines.append("```")
                    else:
                        output_lines.append(passage)
                    output_lines.append("")

        if example.strip():
            s2s_results = nv_retriever.retrieve_s2s([example], top_k=top_k_val)
            if s2s_results:
                output_lines.append("## Top-k for Example (Q2Q) Retrieval")
                retrieved_s2s = s2s_results[0]["retrieved_passages"]
                for rank, (passage, score) in enumerate(retrieved_s2s, start=1):
                    output_lines.append(f"**Rank {rank}** | Score: {score:.4f}")
                    # Check if the passage contains backticks or code-like content
                    if '`' in passage or any(char in passage for char in ['{', '}', ';']):
                        output_lines.append("```")
                        output_lines.append(passage)
                        output_lines.append("```")
                    else:
                        output_lines.append(passage)
                    output_lines.append("")

        return "\n".join(output_lines) if output_lines else "No query or example provided. Please enter at least one."

    def oai_retrieve_fn(query: str, example: str, instruction: str, top_k: int) -> str:
        if instruction and query:
            query = f"{instruction}: {query}"
        return oai_retrieve(query, int(top_k), index, documents)

    # Build the combined interface
    with gr.Blocks() as demo:
        gr.Markdown("# Combined Retrieval Systems Demo")
        
        with gr.Tabs() as tabs:
            # NV-Embed Tab
            with gr.Tab("NV-Embed Retrieval"):
                with gr.Row():
                    with gr.Column(scale=1):
                        nv_query = gr.Textbox(
                            label="Enter your query/prompt description:",
                            placeholder="e.g. 'creative writing questions for generating long nonfiction stories'"
                        )
                        nv_example = gr.Textbox(
                            label="Example problem (optional)",
                            placeholder="e.g. 'a combinatorics question: There are 10 people...' (for Q2Q retrieval)"
                        )
                        nv_instruction = gr.Textbox(
                            label="Custom instruction (optional)",
                            placeholder="e.g. 'Given the description of a type of question, retrieve passages that contain questions of that type'"
                        )
                        nv_top_k = gr.Number(
                            label="Top K",
                            value=5,
                            precision=0
                        )
                        nv_submit = gr.Button("Retrieve (NV-Embed)")
                    with gr.Column(scale=2):
                        nv_output = gr.Markdown(label="Retrieved Passages")

            # OpenAI Tab
            with gr.Tab("OpenAI Retrieval"):
                with gr.Row():
                    with gr.Column(scale=1):
                        oai_query = gr.Textbox(
                            label="Enter your query/prompt description:",
                            placeholder="e.g. 'creative writing questions for generating long nonfiction stories'"
                        )
                        oai_example = gr.Textbox(
                            label="Example problem (optional)",
                            placeholder="e.g. 'a combinatorics question: There are 10 people...' (for Q2Q retrieval)"
                        )
                        oai_instruction = gr.Textbox(
                            label="Custom instruction (optional)",
                            placeholder="e.g. 'Given the description of a type of question, retrieve passages that contain questions of that type'"
                        )
                        oai_top_k = gr.Number(
                            label="Top K",
                            value=5,
                            precision=0
                        )
                        oai_submit = gr.Button("Retrieve (OpenAI)")
                    with gr.Column(scale=2):
                        oai_output = gr.Markdown(label="Retrieved Passages")

        # Set up click events
        nv_submit.click(
            fn=nv_retrieve_fn,
            inputs=[nv_query, nv_example, nv_instruction, nv_top_k],
            outputs=[nv_output]
        )
        
        oai_submit.click(
            fn=oai_retrieve_fn,
            inputs=[oai_query, oai_example, oai_instruction, oai_top_k],
            outputs=[oai_output]
        )

    return demo

if __name__ == "__main__":
    demo = build_combined_demo()
    demo.launch(share=True) 