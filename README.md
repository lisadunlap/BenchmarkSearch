# Eval Set Retrival System (Currently in progress)

This repository contains a lightweight retrieval system designed for retrieving prompt-like passages from a corpus. It uses:

1. A caching mechanism in an LMDB database to store embeddings (to avoid recomputing them).  
2. A FAISS index for efficient approximate nearest-neighbor search.  
3. A Hugging Face Transformer-based model ("nvidia/NV-Embed-v2") for obtaining embeddings.  
4. A Gradio UI to let users quickly interact with the retrieval API in their browser.

---

## Code Overview

### 1. LMDBCache
• Defined in the class LMDBCache.  
• Stores and retrieves embedding tensors keyed by a hash of (instruction + text).  
• Automatically doubles the LMDB map size on a MapFullError.

### 2. RetrievalSystem
• Main class for building the index and running queries.  
• Key functionalities:  
  – encode_texts(...) → splits data into batches, uses multiple threads to speed up encoding, caches embeddings.  
  – build_faiss_index(...) → encodes all passages, adds them to a FAISS index, and saves the passage text list for subsequent retrieval.  
  – retrieve(...) → standard retrieval using a prompt-based instruction.  
  – retrieve_s2s(...) → question-to-question retrieval using a different instruction prefix.  
  – retrieve_combined(...) → new functionality that performs a weighted average of similarity from a main prompt and an optional example input for question-to-question similarity.

### 3. Combined Retrieval (“retrieve_combined”)
This method provides a way to incorporate both:  
• Prompt description similarity (e.g. “creative writing questions...”)  
• Question-to-question (Q2Q) similarity (e.g. “Given a question, retrieve questions of the same type...”)  

It computes two separate FAISS similarity scores (one for the prompt description, one for the Q2Q), then merges them by taking:
final_score = α * (prompt-based score) + (1−α) * (Q2Q-based score).

This means you can tune α:
• α=1 = use only prompt-based score.  
• α=0 = use only Q2Q-based score.  
• α=0.5 = balance them equally.  

### 4. Gradio Interface (build_gradio_app)
• Provides a simple web UI for entering text queries and seeing retrieved passages.  
• By default, it shows the “query” (prompt description) textbox, an “example problem” textbox, a numeric input for top_k, and an α slider controlling the weighting.  
• Launches a local server that you can access in your browser.

---

## Usage Instructions

Follow these steps to get the system up and running:

1. Install Dependencies:  
   • Make sure you have Python 3.8+ installed.  
   • Install packages:  
     pip install torch transformers faiss-cpu gradio tqdm lmdb pandas

   (You may also need CUDA-compatible PyTorch if you plan to run on GPU.)

2. Prepare Input Files:  
   • The script expects two JSON files:  
     – arena-human-preference-55k.json  
     – competition_math_test.json  
   • These files have sample passages/questions. If you have your own data, you can adapt how they’re loaded.

3. Run the Script:  
   • Execute the Python file:  
     python toy_retrival_system2.py  
   • On successful launch, Gradio will provide you with a local URL (e.g., http://127.0.0.1:7860) and an optional shareable link.

4. Interact with the UI:  
   • Type in your “Main prompt description.”  
   • Optionally, add an “Example problem” for Q2Q retrieval.  
   • Adjust “Top K” to change how many results you see.  
   • Adjust “Weight for Query Similarity (alpha)” to control how much weight you give the main prompt vs. example similarity.  
   • Click “Retrieve” to see your results.

### LMDB Notes
• The LMDB database file is named “embedding_cache.lmdb” by default.  
• Embeddings are stored as pickled data keyed by a SHA-256 hash of the text and instruction.  
• If you modify your data or your model, you can remove or rename the existing “embedding_cache.lmdb” so it starts fresh.

### GPU Usage
• The code uses device_map="balanced_low_0", which attempts to distribute the model across available GPUs.  
• If you have fewer GPUs or want to use CPU, you can pass device='cpu' when instantiating the RetrievalSystem.

---

## File Structure

Below is the high-level structure for toy_retrival_system2.py:

1) Imports and dependencies.  
2) LMDBCache class.  
3) RetrievalSystem class:  
   – Initialization and a caching embedding step.  
   – Methods for batching / encoding text (encode_texts).  
   – Building a FAISS index (build_faiss_index).  
   – Retrieving top-k passages with different instructions (retrieve, retrieve_s2s).  
   – Weighted combined retrieval (retrieve_combined)  
4) A Gradio UI constructor (build_gradio_app) with text inputs, sliders, and display areas.  
5) Main execution block:  
   – Load data from JSON files.  
   – Instantiate the retrieval system.  
   – Build the FAISS index over passages.  
   – Launch Gradio.  
   – Close LMDB cache on shutdown.

---

### Potential Future Enhancements
• Add in metadata for each passage (e.g. question type, difficulty, dataset description, etc)
• Add in a way to retrieve passages based on metadata and similarity
• Expand to retrieve passages based on a question and a set of examples
• Add in simple filters (english, length, etc)
• Toy test env which uses existing datasets 
