import json
import os
import argparse
import numpy as np
from datetime import datetime
from dotenv import dotenv_values 
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

from src.agent.real_agent import OpenAICompatibleAgent
from src.monitor.probe import StateMonitor

def calculate_cosine_distance(embedding_model, text1, text2):
    """Calculates semantic distance between two texts."""
    if not text1 or not text2:
        return 1.0
    embeddings = embedding_model.encode([text1, text2])
    return cosine(embeddings[0], embeddings[1])

def run_telephone_game(model_name: str, api_key: str, base_url: str, hops: int = 10):
    print(f"--- Starting Telephone Game Experiment ({hops} hops) ---")
    print(f"Model: {model_name}")
    
    # 1. Setup Components
    agent = OpenAICompatibleAgent(model_name=model_name, api_key=api_key, base_url=base_url, temperature=0.7)
    monitor = StateMonitor()
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. The "Secret" (Ground Truth)
    # A specific, factual instruction that is easy to distort.
    ground_truth = "Critical Protocol: The database backup runs at 03:00 UTC. You must verify the integrity using SHA-256 checksums. If the checksum ends in 'F', retry the backup immediately."
    
    print(f"Ground Truth: {ground_truth}\n")
    
    # 3. Logging Setup
    log_dir = "data/logs_telephone"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/telephone_{timestamp}.jsonl"
    
    current_message = ground_truth
    
    with open(log_file, "w") as f:
        # 4. The Chain Loop
        for hop in range(1, hops + 1):
            print(f"[Hop {hop}] Agent {hop} receiving message...")
            
            # Construct Prompt: "You are Agent X. Pass this info along."
            # We do NOT include previous history. This is a fresh context.
            prompt = f"You are a relay agent. Your task is to accurately pass the following information to the next agent in the chain. Do not add new information, but ensure the instructions are clear.\n\nINFORMATION TO PASS:\n{current_message}"
            
            history = [{"role": "user", "content": prompt}]
            
            # Get Action (Text Generation)
            response = agent.get_next_action(history)
            generated_text = response.get("content", "")
            logprobs = response.get("logprobs", [])
            
            # Metrics
            entropy = monitor.calculate_entropy_from_logprobs(logprobs)
            avg_entropy = entropy / len(logprobs) if logprobs else 0.0
            
            # Semantic Drift (Distance from Ground Truth)
            drift_from_truth = calculate_cosine_distance(embedding_model, ground_truth, generated_text)
            
            # Semantic Shift (Distance from Previous Hop - "Mutation Rate")
            shift_from_prev = calculate_cosine_distance(embedding_model, current_message, generated_text)
            
            # Log Data
            entry = {
                "hop": hop,
                "input_message": current_message,
                "output_message": generated_text,
                "avg_token_entropy": float(avg_entropy),
                "semantic_drift_from_truth": float(drift_from_truth),
                "step_mutation_score": float(shift_from_prev),
                "timestamp": datetime.now().isoformat()
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()
            
            print(f"  > Entropy: {avg_entropy:.4f}")
            print(f"  > Drift from Truth: {drift_from_truth:.4f}")
            print(f"  > Output Start: {generated_text[:50]}...")
            
            # Update message for next agent (The "Handoff")
            current_message = generated_text

    print(f"\n--- Experiment Complete. Data saved to {log_file} ---")

if __name__ == "__main__":
    # Load Config
    config = dotenv_values(".env")
    api_key = config.get("VLLM_API_KEY")
    base_url = config.get("VLLM_BASE_URL")
    model_name = config.get("VLLM_MODEL_NAME", "deepseek-chat")
    
    if not api_key or not base_url:
        print("Error: Missing .env configuration.")
        exit(1)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--hops", type=int, default=10, help="Number of agents in the chain")
    args = parser.parse_args()
    
    run_telephone_game(model_name, api_key, base_url, args.hops)
