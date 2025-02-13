import os
import pickle
import lmdb
import torch
import torch.nn.functional as F
import hashlib
import faiss  # FAISS for vector search
from transformers import AutoModel
import pandas as pd
import ast
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import gradio as gr
from typing import List, Dict, Tuple, Optional
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from data_prep import load_datasets

###############################################################################
# LMDBCache: A class to handle caching embeddings in LMDB with hashed keys.
###############################################################################
class LMDBCache:
    def __init__(self, db_path: str = "embedding_cache.lmdb", map_size: int = 2**32) -> None:
        """
        :param db_path: Path to the LMDB database file.
        :param map_size: Maximum size the database may grow to (in bytes).
        """
        self.db_path = db_path
        self.env = lmdb.open(
            self.db_path,
            map_size=map_size,
            subdir=False,
            lock=True,
            writemap=True,
            create=True
        )

    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Retrieve a cached embedding for a given (hashed) key, or None if not present.
        """
        with self.env.begin(write=False) as txn:
            data = txn.get(key.encode("utf-8"))
            if data is None:
                return None
            else:
                return pickle.loads(data)

    def set(self, key: str, value: torch.Tensor) -> None:
        """
        Store an embedding in the cache under the given (hashed) key.
        Retry with doubled map_size if MapFullError occurs.
        """
        try:
            with self.env.begin(write=True) as txn:
                txn.put(key.encode("utf-8"), pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except lmdb.MapFullError:
            # Double the map size and retry
            current_size = self.env.info()['map_size']
            new_size = current_size * 2
            self.env.set_mapsize(new_size)
            # Retry the operation
            with self.env.begin(write=True) as txn:
                txn.put(key.encode("utf-8"), pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))

    def close(self) -> None:
        """
        Close the LMDB environment (should be called on shutdown).
        """
        self.env.close()


###############################################################################
# RetrievalSystem class with FAISS indexing.
###############################################################################
class RetrievalSystem:
    def __init__(self, 
                 model_name: str = 'nvidia/NV-Embed-v2', 
                 lmdb_path: str = "embedding_cache.lmdb", 
                 map_size: int = 2**30,
                 use_faiss: bool = True,
                 device: str = 'cuda',
                 num_gpus: int = 2) -> None:
        """
        :param model_name: Name of the embedding model to load.
        :param lmdb_path: Path to the LMDB file for caching embeddings.
        :param map_size: Size of the LMDB map in bytes (default ~1GB).
        :param use_faiss: Whether to build/use FAISS for retrieval.
        :param device: Device to use for computation ('cuda' or 'cpu').
        :param num_gpus: Number of GPUs to use (just a reference here).
        """
        self.cache = LMDBCache(db_path=lmdb_path, map_size=map_size)

        # Decide device string, but the multi-GPU logic is in device_map below.
        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        #######################################################################
        # Load from the HF Hub with device_map="balanced_low_0".
        # This instructs Transformers to shard the model across all visible GPUs.
        #######################################################################
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="balanced_low_0",  # tries to distribute across GPUs
            low_cpu_mem_usage=True        # can help reduce CPU RAM usage
        )

        # FAISS-related variables
        self.use_faiss = use_faiss
        self.faiss_index: Optional[faiss.Index] = None
        self.passage_texts: Optional[List[str]] = None
        self.embedding_dim: Optional[int] = None

    def _make_key(self, instruction: str, text: str) -> str:
        """
        Hash (instruction + text) to produce a short, safe key for LMDB.
        Handle invalid Unicode characters by replacing them.
        """
        cleaned_instruction = instruction.encode('utf-8', errors='replace').decode('utf-8')
        cleaned_text = text.encode('utf-8', errors='replace').decode('utf-8')
        
        raw_key = (cleaned_instruction + "\n" + cleaned_text).encode("utf-8")
        hashed = hashlib.sha256(raw_key).hexdigest()
        return hashed

    def _process_batch(
        self, 
        batch: List[str], 
        instruction: str, 
        normalize: bool
    ) -> torch.Tensor:
        """
        Process a single batch of texts to produce embeddings.
        """
        batch_embeddings = []
        for text in batch:
            key = self._make_key(instruction, text)
            cached_emb = self.cache.get(key)
            if cached_emb is not None:
                emb = cached_emb
            else:
                # Mixed precision context if on GPU
                if self.device.startswith("cuda"):
                    with torch.cuda.amp.autocast():
                        emb = self.model.encode([text], instruction=instruction)
                else:
                    emb = self.model.encode([text], instruction=instruction)

                # Convert to CPU for caching
                emb = emb.detach().cpu().numpy()[0]
                self.cache.set(key, emb)
            batch_embeddings.append(emb)
            
        batch_embeddings_array = np.array(batch_embeddings)
        batch_embeddings_tensor = torch.from_numpy(batch_embeddings_array).float()
        if normalize:
            batch_embeddings_tensor = F.normalize(batch_embeddings_tensor, p=2, dim=1)
        return batch_embeddings_tensor

    def encode_texts(
        self, 
        texts: List[str], 
        instruction: str = "", 
        batch_size: int = 8, 
        normalize: bool = True, 
        max_workers: int = 16
    ) -> torch.Tensor:
        """
        Encode texts using parallel processing with ThreadPoolExecutor and LMDB caching.
        Preserves the order of inputs by placing the results of each future back 
        into the correct position before concatenating.
        """
        # Split texts into batches
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        results = [None] * len(batches)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_map = {}
            # Submit each batch to the thread pool, storing which index it belongs to
            for batch_idx, batch in enumerate(batches):
                future = executor.submit(self._process_batch, batch, instruction, normalize)
                futures_map[future] = batch_idx

            # As they complete, store them in the correct slot
            for future in tqdm(concurrent.futures.as_completed(futures_map), total=len(futures_map), desc="Processing batches"):
                batch_idx = futures_map[future]
                results[batch_idx] = future.result()

        # Now concatenate all in the original order
        all_embeddings_tensor = torch.cat(results, dim=0)
        return all_embeddings_tensor

    def build_faiss_index(
        self, 
        passages: List[str], 
        instruction: str = "", 
        batch_size: int = 64, 
        normalize: bool = True, 
        max_workers: int = 4
    ) -> None:
        """
        Build a FAISS index for fast retrieval, and also store 
        the passage embeddings locally for combined-scoring retrieval.
        """
        print("Building FAISS index...")
        self.passage_texts = passages

        # Get embeddings for passages
        passage_embs = self.encode_texts(
            passages, 
            instruction=instruction, 
            batch_size=batch_size, 
            normalize=normalize,
            max_workers=max_workers
        )
        passage_embs_np = passage_embs.cpu().numpy()
        self.embedding_dim = passage_embs.shape[1]

        # Keep a copy of these embeddings around for custom scoring
        self.passage_embs_np = passage_embs_np

        # IndexFlatIP (exact search)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(passage_embs_np)

        self.faiss_index = index
        print(f"FAISS index built: {index.ntotal} vectors, dim={self.embedding_dim}.")

    def retrieve(
        self, 
        query_texts: List[str], 
        top_k: int = 3,
        instruction: str = None
    ) -> List[Dict[str, List[Tuple[str, float]]]]:
        """
        For each query in `query_texts`, retrieve the top_k most similar passages.
        Uses FAISS if `self.use_faiss` is True, otherwise does a naive dot-product search.

        :param query_texts: List of query strings.
        :param top_k: Number of top matches to return.
        :param instruction: Optional custom instruction for embedding.
        :return: List of dictionaries, each with keys:
                 {
                    "query": str,
                    "retrieved_passages": [(passage: str, score: float), ...]
                 }
        """
        # Use custom instruction if provided, otherwise use default
        if instruction:
            query_prefix = f"Instruct: {instruction}\nQuery: "
        else:
            task_name_to_instruct = {
                "question": "Given the description of a type of question, retrieve passages that contain questions of that type",
                "task_based": "Given the description of a LLM task, retrieve prompts which could be used to evaluate the task",
            }
            query_prefix = "Instruct: " + task_name_to_instruct["question"] + "\nQuery: "
        
        # Encode queries
        query_embs = self.encode_texts(query_texts, instruction=query_prefix)

        if self.use_faiss and self.faiss_index is not None and self.passage_texts is not None:
            query_embs_np = query_embs.cpu().numpy()

            # FAISS search
            scores, indices = self.faiss_index.search(query_embs_np, top_k)

            results = []
            for i, query in enumerate(query_texts):
                retrieved_passages: List[Tuple[str, float]] = []
                for j in range(top_k):
                    doc_idx = indices[i, j]
                    score_val = float(scores[i, j])
                    doc_text = self.passage_texts[doc_idx]
                    retrieved_passages.append((doc_text, score_val))

                results.append({
                    "query": query,
                    "retrieved_passages": retrieved_passages
                })

            return results
        else:
            raise ValueError("FAISS index not built or use_faiss=False. Please build the index first.")

    def retrieve_s2s(
        self, 
        query_texts: List[str], 
        top_k: int = 3
    ) -> List[Dict[str, List[Tuple[str, float]]]]:
        """
        Perform a question-to-question (Q2Q) retrieval using a different instruction prefix.
        This is used to find questions that are of the same type or style as the provided query.
        
        :param query_texts: List of query questions.
        :param top_k: Number of top matches to return.
        :return: List of dictionaries, each with keys:
                 {
                    "query": str,
                    "retrieved_passages": [(passage: str, score: float), ...]
                 }
        """
        if not self.use_faiss or self.faiss_index is None or self.passage_texts is None:
            raise ValueError("FAISS index not built or use_faiss=False. Please build the index first.")

        # Use a different prefix for Q2Q retrieval
        q2q_prefix = "Given a question, retrieve questions that are of the same question type.\nQuery: "
        query_embs = self.encode_texts(query_texts, instruction=q2q_prefix)

        query_embs_np = query_embs.cpu().numpy()

        # FAISS search
        scores, indices = self.faiss_index.search(query_embs_np, top_k)

        results = []
        for i, query in enumerate(query_texts):
            retrieved_passages: List[Tuple[str, float]] = []
            for j in range(top_k):
                doc_idx = indices[i, j]
                score_val = float(scores[i, j])
                doc_text = self.passage_texts[doc_idx]
                retrieved_passages.append((doc_text, score_val))
            results.append({
                "query": query,
                "retrieved_passages": retrieved_passages
            })

        return results

    def retrieve_combined(
        self,
        query_text: Optional[str],
        example_text: Optional[str],
        top_k: int = 3,
        alpha: float = 0.5
    ) -> List[Dict[str, List[Tuple[str, float]]]]:
        """
        Perform a retrieval that combines:
         - Prompt-based similarity (using 'query_text')
         - S2S (question-to-question) similarity (using 'example_text')
         
        via a weighted sum of their scores (alpha for query, 1-alpha for example).
        
        If only one text is supplied, it defaults to that method alone.
        
        :param query_text: The main 'prompt description' query.
        :param example_text: An example question for Q2Q retrieval.
        :param top_k: How many results to return.
        :param alpha: Weight for the main query. (1 - alpha) is weight for the Q2Q similarity.
        :return: A single ranked list reflecting the combined similarity.
        """
        if not self.use_faiss or self.faiss_index is None or self.passage_texts is None:
            raise ValueError("FAISS index not built or use_faiss=False. Please build the index first.")

        # Edge-case: If both are empty, there's nothing to do
        if (not query_text) and (not example_text):
            return [{
                "query": "",
                "retrieved_passages": []
            }]

        # Compute query embedding if query_text is present
        if query_text:
            # The same logic used in retrieve():
            task_name_to_instruct = {
                "question": "Given the description of a type of question, retrieve passages that contain questions of that type",
            }
            query_prefix = "Instruct: " + task_name_to_instruct["question"] + "\nQuery: "
            q_emb = self.encode_texts([query_text], instruction=query_prefix)
            q_scores, _ = self.faiss_index.search(q_emb.cpu().numpy(), len(self.passage_texts))
            # shape: [batch=1, #passages]
            q_scores = q_scores[0]  # The single query
        else:
            # No query => zero array
            q_scores = np.zeros(len(self.passage_texts), dtype=np.float32)

        # Compute example embedding if example_text is present
        if example_text:
            q2q_prefix = "Given a question, retrieve questions that are of the same question type.\nQuery: "
            e_emb = self.encode_texts([example_text], instruction=q2q_prefix)
            e_scores, _ = self.faiss_index.search(e_emb.cpu().numpy(), len(self.passage_texts))
            e_scores = e_scores[0]
        else:
            e_scores = np.zeros(len(self.passage_texts), dtype=np.float32)

        # Weighted sum of both similarities
        # alpha * (prompt-based) + (1 - alpha) * (Q2Q-based)
        final_scores = alpha * q_scores + (1 - alpha) * e_scores

        # Get top_k by sorting
        ranked_indices = np.argsort(-final_scores)  # descending
        top_indices = ranked_indices[:top_k]

        # Prepare final results
        retrieved_passages: List[Tuple[str, float]] = []
        for i in top_indices:
            retrieved_passages.append((
                self.passage_texts[i],
                float(final_scores[i])
            ))

        # Return a single result object
        return [{
            "query": f"MAIN: {query_text} | EXAMPLE: {example_text}",
            "retrieved_passages": retrieved_passages
        }]

    def close_cache(self) -> None:
        """
        Close the LMDB cache environment when done.
        """
        self.cache.close()


###############################################################################
# Gradio App
###############################################################################
def build_gradio_app(retriever: RetrievalSystem) -> gr.Blocks:
    """
    Build and return a Gradio Blocks interface for the FAISS-based retrieval system,
    now with optional "Example problem" and "Custom instruction" inputs.
    """
    def retrieve_fn(query: str, example: str, instruction: str, top_k: int) -> str:
        """
        Retrieve results using optional custom instruction for the main query.
        """
        # Convert top_k to int safely
        try:
            top_k_val = int(top_k)
        except ValueError:
            top_k_val = 3

        output_lines = []

        # 1) Standard retrieval using "prompt description" if available
        if query.strip():
            results = retriever.retrieve([query], top_k=top_k_val, instruction=instruction)
            if results:
                output_lines.append("## Top-k for Prompt Description Retrieval")
                retrieved = results[0]["retrieved_passages"]
                for rank, (passage, score) in enumerate(retrieved, start=1):
                    output_lines.append(f"**Rank {rank}** | Score: {score:.4f}")
                    output_lines.append(passage)
                    output_lines.append("")  # blank line
            else:
                output_lines.append("No results returned for your main prompt description query.")

        # 2) Question-to-question retrieval if an example is provided
        if example.strip():
            s2s_results = retriever.retrieve_s2s([example], top_k=top_k_val)
            if s2s_results:
                # Add a section heading
                output_lines.append("## Top-k for Example (Q2Q) Retrieval")
                retrieved_s2s = s2s_results[0]["retrieved_passages"]
                for rank, (passage, score) in enumerate(retrieved_s2s, start=1):
                    output_lines.append(f"**Rank {rank}** | Score: {score:.4f}")
                    output_lines.append(passage)
                    output_lines.append("")  # blank line
            else:
                output_lines.append("No Q2Q results returned for your example query.")

        if not output_lines:
            return "No query or example provided. Please enter at least one."
        else:
            return "\n".join(output_lines)

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

        submit_btn.click(fn=retrieve_fn, 
                        inputs=[query_input, example_input, instruction_input, top_k_input], 
                        outputs=[output_box])

    return demo


if __name__ == "__main__":
    # 1) Load data
    dataset_names = ['arena', 'math', 'narrativeqa', 'movies']
    passages = load_datasets(dataset_names)

    # 2) Initialize the retrieval system
    #    We use device_map="balanced_low_0" behind the scenes to shard across GPUs
    retriever = RetrievalSystem(
        model_name='nvidia/NV-Embed-v2',
        lmdb_path="embedding_cache.lmdb",
        map_size=2**32,
        use_faiss=True,
        device='cuda',
        num_gpus=2
    )

    # 3) Build FAISS index over the passages
    retriever.build_faiss_index(passages, batch_size=32)

    # 4) Build Gradio interface
    demo = build_gradio_app(retriever)
    demo.launch(share=True)

    # 5) (Optional) Close LMDB after shutting down the Gradio app
    retriever.close_cache()