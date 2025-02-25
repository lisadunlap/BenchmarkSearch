import json
import logging
import os
import threading
from typing import List
import concurrent.futures
from tqdm import tqdm

import lmdb
import openai
from openai import OpenAI
import anthropic
import datetime
import numpy as np
from serve.embedding_client import get_text_embedding
logging.basicConfig(level=logging.ERROR)

if not os.path.exists("cache/llm_cache"):
    os.makedirs("cache/llm_cache")

if not os.path.exists("cache/llm_embed_cache"):
    os.makedirs("cache/llm_embed_cache")

llm_cache = lmdb.open("cache/llm_cache", map_size=int(1e11))
llm_embed_cache = lmdb.open("cache/llm_embed_cache", map_size=int(1e11))

import hashlib
from typing import Dict, List, Optional

import lmdb
from PIL import Image
import pickle
import requests


def hash_key(key) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def get_from_cache(key: str, env: lmdb.Environment) -> Optional[str]:
    with env.begin(write=False) as txn:
        hashed_key = hash_key(key)
        value = txn.get(hashed_key.encode())
    if value:
        return value.decode()
    return None


def save_to_cache(key: str, value: str, env: lmdb.Environment):
    with env.begin(write=True) as txn:
        hashed_key = hash_key(key)
        txn.put(hashed_key.encode(), value.encode())


def save_emb_to_cache(key: str, value, env: lmdb.Environment):
    with env.begin(write=True) as txn:
        hashed_key = hash_key(key)
        # Use pickle to serialize the value
        serialized_value = pickle.dumps(value)
        txn.put(hashed_key.encode(), serialized_value)


def get_emb_from_cache(key: str, env: lmdb.Environment):
    with env.begin(write=False) as txn:
        hashed_key = hash_key(key)
        serialized_value = txn.get(hashed_key.encode())
        if serialized_value is not None:
            # Deserialize the value back into a Python object
            value = pickle.loads(serialized_value)
            return value
        else:
            # Handle the case where the key does not exist in the cache
            return None


def get_llm_output(
    prompt: str | List[str], model: str, cache=True, system_prompt=None, history=[], max_tokens=256
) -> str | List[str]:
    # Handle list of prompts with thread pool
    if isinstance(prompt, list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(
                    get_llm_output, p, model, cache, system_prompt, history, max_tokens
                )
                for p in prompt
            ]
            return [f.result() for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures))]

    # Original single prompt logic
    openai.api_base = (
        "https://api.openai.com/v1" if model != "llama-3-8b" else "http://localhost:8001/v1"
    )
    if "gpt" in model:
        client = OpenAI()
    elif model == "llama-3-8b":
        client = OpenAI(
            base_url="http://localhost:8001/v1",
        )
    else:
        client = anthropic.Anthropic()

    systems_prompt = (
        "You are a helpful assistant." if not system_prompt else system_prompt
    )

    if "gpt" in model:
        messages = (
            [{"role": "system", "content": systems_prompt}]
            + history
            + [
                {"role": "user", "content": prompt},
            ]
        )
    elif "claude" in model:
        messages = history + [
            {"role": "user", "content": prompt},
        ]
    else:
        # messages = prompt
        messages = (
            [{"role": "system", "content": systems_prompt}]
            + history
            + [
                {"role": "user", "content": prompt},
            ]
        )
    key = json.dumps([model, messages])

    cached_value = get_from_cache(key, llm_cache) if cache else None
    if cached_value is not None:
        logging.debug(f"LLM Cache Hit")
        return cached_value
    else:
        logging.debug(f"LLM Cache Miss")

    for _ in range(3):
        try:
            if "gpt-3.5" in model:
                start_time_ms = datetime.datetime.now().timestamp() * 1000
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                end_time_ms = round(
                    datetime.datetime.now().timestamp() * 1000
                )  # logged in milliseconds
                response = completion.choices[0].message.content.strip()
            elif "gpt-4" in model:
                start_time_ms = datetime.datetime.now().timestamp() * 1000
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                end_time_ms = round(
                    datetime.datetime.now().timestamp() * 1000
                )  # logged in milliseconds
                response = completion.choices[0].message.content.strip()
            elif "claude-opus" in model:
                completion = client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024,
                    system=systems_prompt,
                )
                response = completion.content[0].text
            elif "claude" in model:
                completion = client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                response = completion.content[0].text
            elif model == "vicuna":
                completion = client.chat.completions.create(
                    model="lmsys/vicuna-7b-v1.5",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,  # TODO: greedy may not be optimal
                )
                response = completion.choices[0].message.content.strip()
            elif model == "llama-3-8b":
                completion = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    messages=messages,
                    max_tokens=max_tokens,
                    extra_body={"stop_token_ids": [128009]},
                )
                response = (
                    completion.choices[0]
                    .message.content.strip()
                    .replace("<|eot_id|>", "")
                )

            save_to_cache(key, response, llm_cache)
            return response

        except Exception as e:
            logging.error(f"LLM Error: {e}")
            # if error is Error Code: 400, then it is likely that the prompt is too long, so truncate it
            if "Error code: 400" in str(e):
                messages = (
                    [{"role": "system", "content": systems_prompt}]
                    + history
                    + [
                        {"role": "user", "content": prompt[: int(len(prompt) / 2)]},
                    ]
                )
            else:
                raise e
    return "LLM Error: Cannot get response."


def get_llm_embedding(prompt: str | List[str], model: str, instruction: str = "", cache=True) -> str | List[str]:
    # Handle list of prompts with thread pool
    if isinstance(prompt, list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(get_llm_embedding, p, model, instruction, cache)
                for p in prompt
            ]
            return [f.result() for f in concurrent.futures.as_completed(futures)]

    # Original single prompt logic
    openai.api_base = "https://api.openai.com/v1"
    client = OpenAI()
    key = json.dumps([model, prompt, instruction])

    cached_value = get_emb_from_cache(key, llm_embed_cache) if cache else None

    if cached_value is not None:
        logging.debug(f"LLM Embedding Cache Hit")
        # normalize the embedding
        cached_value = cached_value / np.linalg.norm(cached_value)
        return cached_value
    else:
        logging.debug(f"LLM Embedding Cache Miss")

    # Fallback to original method if server fails or model is not NVEmbedV2
    for _ in range(3):
        try:
            # Use the Flask embedding server if the model is NVEmbedV2
            if model == "nvidia/NV-Embed-v2":
                embedding = get_text_embedding([prompt], instruction, server_url="http://localhost:5000")[0]
            else:
                text = prompt.replace("\n", " ")
                embedding = (
                    client.embeddings.create(input=[text], model=model).data[0].embedding
                )
            save_emb_to_cache(key, embedding, llm_embed_cache)
            # normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            logging.error(f"LLM Error: {e}")
            continue

    return None


def test_get_llm_output():
    prompt = "hello"
    model = "gpt-4"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "gpt-3.5-turbo"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "vicuna"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")

def test_get_llm_embedding():
    prompt = "hello"
    model = "nvidia/NV-Embed-v2"
    embedding = get_llm_embedding(prompt, model, instruction="", cache=False)
    print(f"{model=}, {np.array(embedding).shape}")
    # normalize the embedding
    embedding = embedding / np.linalg.norm(embedding)
    # test the embedding server# ensure the embedding is correct (see if the embedding is close to the original prompt but far from other prompts)
    prompt2 = "hello"
    far_prompt = "what is the capital of the moon?"
    middle_prompt = "how are you?"
    embedding2 = get_llm_embedding(prompt2, model, cache=False)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    print(f"{model=}, {np.array(embedding2).shape}")
    print(f"cosine similarity (should be close to 1): {np.dot(embedding, embedding2) / (np.linalg.norm(embedding) * np.linalg.norm(embedding2))}")
    far_embedding = get_llm_embedding(far_prompt, model, cache=False)
    far_embedding = far_embedding / np.linalg.norm(far_embedding)
    print(f"cosine similarity (low): {np.dot(embedding, far_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(far_embedding))}")
    
    
    model = "text-embedding-3-small"
    embedding = get_llm_embedding(prompt, model, cache=False)
    print(f"{model=}, {np.array(embedding).shape}")
    embedding = embedding / np.linalg.norm(embedding)
    embedding2 = get_llm_embedding(middle_prompt, model, cache=False)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    print(f"cosine similarity (medium): {np.dot(embedding, embedding2) / (np.linalg.norm(embedding) * np.linalg.norm(embedding2))}")

    # ensure the embedding is correct (see if the embedding is close to the original prompt but far from other prompts)
    prompt2 = "hello"
    far_prompt = "what is the capital of the moon?"
    middle_prompt = "how are you?"
    embedding2 = get_llm_embedding(prompt2, model, cache=False)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    print(f"{model=}, {np.array(embedding2).shape}")
    print(f"cosine similarity (should be close to 1): {np.dot(embedding, embedding2) / (np.linalg.norm(embedding) * np.linalg.norm(embedding2))}")
    far_embedding = get_llm_embedding(far_prompt, model, cache=False)
    far_embedding = far_embedding / np.linalg.norm(far_embedding)
    print(f"cosine similarity: {np.dot(embedding, far_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(far_embedding))}")
    embedding = embedding / np.linalg.norm(embedding)
    embedding2 = get_llm_embedding(middle_prompt, model, cache=False)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    print(f"cosine similarity (medium): {np.dot(embedding, embedding2) / (np.linalg.norm(embedding) * np.linalg.norm(embedding2))}")

if __name__ == "__main__":
    # test_get_llm_output()
    test_get_llm_embedding()