import os
import random
import numpy as np
import pandas as pd
from PIL import Image

def setup_environment(base_path="data/sandbox_task_1"):
    """
    Generates a synthetic defect detection dataset.
    Phase 1: 64x64 Grayscale (Simple)
    Phase 2: 128x128 RGB + Noise (Hard)
    """
    os.makedirs(base_path, exist_ok=True)
    print(f"Generating Vision Dataset at {base_path}...")
    
    dataset_path = os.path.join(base_path, "dataset")
    os.makedirs(os.path.join(dataset_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "test"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "test_new"), exist_ok=True)
    
    labels = []
    
    # --- Phase 1 Generator (Simple) ---
    def generate_image(folder, idx, is_defect, size=(64, 64), mode="L"):
        img = Image.new(mode, size, color=random.randint(50, 200))
        pixels = img.load()
        
        # Draw random "structure" (lines)
        for _ in range(5):
            x = random.randint(0, size[0])
            for y in range(size[1]):
                pixels[x, y] = 255 if mode=="L" else (255, 255, 255)
                
        # Defect: Dark Scratch
        if is_defect:
            x_start = random.randint(10, size[0]-10)
            y_start = random.randint(10, size[1]-10)
            for i in range(10):
                pixels[x_start+i, y_start+i] = 0 if mode=="L" else (0, 0, 0)
                
        filename = f"img_{idx:04d}.png"
        img.save(os.path.join(dataset_path, folder, filename))
        return filename

    # Train Set (100 images)
    for i in range(100):
        is_defect = random.random() < 0.5
        fname = generate_image("train", i, is_defect)
        labels.append({"filename": fname, "label": 1 if is_defect else 0, "split": "train"})
        
    # Test Set Phase 1 (20 images)
    for i in range(20):
        is_defect = random.random() < 0.5
        fname = generate_image("test", i, is_defect)
        labels.append({"filename": fname, "label": 1 if is_defect else 0, "split": "test"})
        
    # --- Phase 2 Generator (Hard - The Shock) ---
    for i in range(20):
        is_defect = random.random() < 0.5
        # 128x128 RGB
        fname = generate_image("test_new", i, is_defect, size=(128, 128), mode="RGB")
        
        # Add Noise (Simulate firmware issue)
        img_path = os.path.join(dataset_path, "test_new", fname)
        img = Image.open(img_path)
        arr = np.array(img)
        noise = np.random.normal(0, 25, arr.shape).astype(np.uint8)
        arr = np.clip(arr + noise, 0, 255)
        Image.fromarray(arr).save(img_path)
        
        labels.append({"filename": fname, "label": 1 if is_defect else 0, "split": "test_new"})

    # Save Labels
    pd.DataFrame(labels).to_csv(os.path.join(dataset_path, "labels.csv"), index=False)
    
    # Save Ground Truth for Oracle (hidden)
    pd.DataFrame(labels).to_csv(os.path.join(base_path, "ground_truth.csv"), index=False)
    
    print("Dataset Generated.")

if __name__ == "__main__":
    setup_environment()
