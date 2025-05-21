import os
import time
import yaml
import boto3
import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv
from scipy.spatial.distance import cosine
from openai import OpenAI
from openai import OpenAIError, RateLimitError

# --- LOAD CONFIG & ENV ---
load_dotenv()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Input mode: local or s3
INPUT_SOURCE = config["input"]["source"]
LOCAL_FILE_PATH = config["input"]["local_path"]

# S3 Config
S3_BUCKET = config["s3"]["bucket"]
S3_KEY = config["s3"]["key"]
S3_LOCAL_PATH = config["s3"]["local_path"]

# Models
LLM_MODEL = config["models"]["llm_model"]
EMBEDDING_MODEL = config["models"]["embedding_model"]

# Runtime
MAX_RETRIES = config["runtime"]["max_retries"]
RETRY_DELAY = config["runtime"]["retry_delay"]
SCORING_STRATEGY = config["runtime"]["scoring_strategy"]

# Prompts
PROMPT_TEMPLATES = config["prompts"]["templates"]

# MLflow
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

# OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- S3 DOWNLOAD ---
def download_from_s3():
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, S3_KEY, S3_LOCAL_PATH)
    print(f"Downloaded {S3_KEY} to {S3_LOCAL_PATH}")
    return S3_LOCAL_PATH

# --- LLM COMPLETION ---
def call_openai_with_retries(prompt, model=LLM_MODEL):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except (OpenAIError, RateLimitError) as e:
            print(f"OpenAI API error (attempt {attempt}): {e}")
            if attempt == MAX_RETRIES:
                return f"[ERROR]: {e}"
            time.sleep(RETRY_DELAY)

# --- LLM SCORING ---
def judge_summary_quality(text, summary, reference=None):
    if reference:
        prompt = (
            "Score the following summary from 1 to 10 based on how well it matches the reference:\n\n"
            f"Original Text:\n{text}\n\n"
            f"Reference Summary:\n{reference}\n\n"
            f"Candidate Summary:\n{summary}\n\n"
            "Score (just the number):"
        )
    else:
        prompt = (
            "Score the following summary from 1 to 10 based on its clarity and accuracy:\n\n"
            f"Original Text:\n{text}\n\n"
            f"Candidate Summary:\n{summary}\n\n"
            "Score (just the number):"
        )
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful evaluator."},
                {"role": "user", "content": prompt}
            ]
        )
        score = response.choices[0].message.content.strip()
        return float(score)
    except Exception as e:
        print(f"[Judge Error] {e}")
        return None

# --- EMBEDDING SCORING ---
def get_embedding(text, model=EMBEDDING_MODEL):
    try:
        response = client.embeddings.create(model=model, input=[text])
        return response.data[0].embedding
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None

def score_with_embedding(text, summary):
    emb_text = get_embedding(text)
    emb_summary = get_embedding(summary)
    if emb_text is None or emb_summary is None:
        return None
    return 1 - cosine(emb_text, emb_summary)

# --- SCORE ROUTER ---
def score_summary(text, summary, reference=None):
    if SCORING_STRATEGY == "llm":
        return judge_summary_quality(text, summary, reference)
    elif SCORING_STRATEGY == "embedding":
        return score_with_embedding(text, summary)
    else:
        raise ValueError(f"Unsupported scoring strategy: {SCORING_STRATEGY}")

# --- SUMMARIZE WITH PROMPT OPTIMIZATION ---
def summarize_text_with_optimization(text, templates, reference_summary=None):
    best_summary = None
    best_template = None
    best_score = -1

    for template in templates:
        prompt = template.format(text=text)
        summary = call_openai_with_retries(prompt)
        summary_length = len(summary) if summary else 1e6
        score = score_summary(text, summary, reference=reference_summary)

        with mlflow.start_run(nested=True):
            mlflow.log_param("prompt_template", template)
            mlflow.log_metric("summary_length", summary_length)
            if score is not None:
                mlflow.log_metric("score", score)
            mlflow.log_text(text, "input.txt")
            mlflow.log_text(summary, "summary.txt")

        if score is not None and score > best_score:
            best_summary = summary
            best_template = template
            best_score = score

    return best_summary, best_template

# --- MAIN PIPELINE ---
def run_pipeline():
    # Load data
    if INPUT_SOURCE == "local":
        file_path = LOCAL_FILE_PATH
    else:
        file_path = download_from_s3()

    df = pd.read_csv(file_path)

    summaries = []
    best_prompts = []

    with mlflow.start_run(run_name="summarization_run"):
        mlflow.log_param("scoring_strategy", SCORING_STRATEGY)
        mlflow.log_param("llm_model", LLM_MODEL)
        mlflow.log_param("embedding_model", EMBEDDING_MODEL)

        for _, row in df.iterrows():
            text = row.get("text", "")
            reference = row.get("reference_summary", None)
            summary, best_prompt = summarize_text_with_optimization(
                text, PROMPT_TEMPLATES, reference_summary=reference
            )
            summaries.append(summary)
            best_prompts.append(best_prompt)

        df["summary"] = summaries
        df["best_prompt"] = best_prompts
        df.to_csv("summarized_output.csv", index=False)
        mlflow.log_artifact("summarized_output.csv")

if __name__ == "__main__":
    run_pipeline()
