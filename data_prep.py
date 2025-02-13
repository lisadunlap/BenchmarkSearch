from datasets import load_dataset
import pandas as pd
import ast
from typing import List, Dict

def load_arena_dataset() -> pd.DataFrame:
    """Load and prepare the Arena dataset."""
    df = load_dataset("lmarena-ai/arena-human-preference-55k")
    df = pd.DataFrame(df['train'])
    df['prompt'] = df['prompt'].apply(ast.literal_eval)
    df['question'] = df['prompt'].apply(lambda x: x[0])
    print(f"Loaded {len(df)} Arena questions")
    return df[['question']]

def load_competition_math() -> pd.DataFrame:
    """Load and prepare the Competition Math dataset."""
    ds = load_dataset("hendrycks/competition_math", trust_remote_code=True)
    math_df = pd.DataFrame(ds['test'])
    math_df['question'] = math_df['problem']
    print(f"Loaded {len(math_df)} Competition Math questions")
    return math_df[['question']]

def load_narrativeqa() -> pd.DataFrame:
    """Load and prepare the NarrativeQA dataset."""
    ds = load_dataset("deepmind/narrativeqa")
    narrativeqa_df = pd.DataFrame(ds['test'])
    narrativeqa_df['question'] = narrativeqa_df['question'].apply(lambda x: x['text'])
    print(f"Loaded {len(narrativeqa_df)} NarrativeQA questions")
    return narrativeqa_df[['question']]

def load_movies() -> pd.DataFrame:
    """Load and prepare the Movies dataset."""
    df = pd.read_json("/home/lisabdunlap/DatasetSearch/data/movies/movies.jsonl", lines=True)
    df['question'] = df.apply(lambda row: f"Movie Title: {row['title']}\nMovie Description: {row['overview']}", axis=1)
    print(f"Loaded {len(df)} movies")
    return df[['question']]

# Dictionary mapping dataset names to their load functions
DATASET_LOADERS = {
    'arena': load_arena_dataset,
    'math': load_competition_math,
    'narrativeqa': load_narrativeqa,
    'movies': load_movies
}

def load_datasets(dataset_names: List[str]) -> List[str]:
    """
    Load and combine multiple datasets.
    
    Args:
        dataset_names: List of dataset names to load
        
    Returns:
        List of questions from all requested datasets
    """
    questions = []
    
    for name in dataset_names:
        if name not in DATASET_LOADERS:
            raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(DATASET_LOADERS.keys())}")
        
        df = DATASET_LOADERS[name]()
        questions.extend(df['question'].tolist())
    
    print(f"Loaded {len(questions)} questions from {dataset_names}")
    return questions
