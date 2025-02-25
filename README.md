# Eval Set Retrival System (Currently in progress)

A lil' retrival system that I am playing around with.

## Setup

```bash
pip install -r requirements.txt
```

## Run NVEmbedding Server 

Will at some point add more models.

```bash
python embedding_server.py
```

## Run Retrival System for a single query

```bash
python toy_retrieval_system.py --dataset vibecheck --query sarcastic --embedding_model text-embedding-3-small
```

## Run Gradio App

```bash
python gradio_app.py
```
