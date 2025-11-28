import os
import random
import string
from src.scenarios.tasks import data_pipeline
from src.scenarios.tasks import vision_dataset

def setup_drug_filter(base_path: str):
    """Creates the environment for the Drug Filter scenario."""
    os.makedirs(base_path, exist_ok=True)
    
    csv_content = "drug_name,weight,solubility,cost\nA,100,0.5,10\nB,150,0.7,15\nC,200,0.3,20\nD,120,0.8,5\nE,250,0.1,50"
    with open(os.path.join(base_path, "drugs.csv"), "w") as f:
        f.write(csv_content)
    print(f"Setup: Created drugs.csv in {base_path}")

def setup_file_organizer(base_path: str):
    """Creates the environment for the File Organizer scenario."""
    if os.path.exists(base_path):
        # Clean up existing files to ensure fresh start
        for f in os.listdir(base_path):
            fp = os.path.join(base_path, f)
            if os.path.isfile(fp):
                os.unlink(fp)
    
    os.makedirs(base_path, exist_ok=True)
    
    print(f"Setup: Generating random files in {base_path}...")
    extensions = ["jpg", "png", "txt"]
    for i in range(20): # 20 files for speed
        ext = extensions[i % 3]
        filename = f"file_{i:03d}_{''.join(random.choices(string.ascii_lowercase, k=4))}.{ext}"
        filepath = os.path.join(base_path, filename)
        with open(filepath, "w") as f:
            f.write(f"Content for {filename}")
            
    print("Setup: Environment Ready.")

def setup_data_pipeline(base_path: str):
    """Creates the environment for the Data Pipeline scenario."""
    data_pipeline.setup_environment(base_path)

def setup_vision_dataset(base_path: str):
    """Creates the environment for the Vision Defect scenario."""
    vision_dataset.setup_environment(base_path)

SCENARIO_SETUP_MAP = {
    "drug_filter_baseline": setup_drug_filter,
    "drug_filter_shock": setup_drug_filter,
    "file_organizer_shock": setup_file_organizer,
    "data_pipeline_shock": setup_data_pipeline,
    "vision_defect_shock": setup_vision_dataset
}
