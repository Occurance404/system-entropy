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

def setup_legacy_refactor(base_path: str):
    """Creates the environment for the Legacy Refactoring challenge."""
    os.makedirs(base_path, exist_ok=True)
    
    # 1. Create the CSV data
    csv_content = "name,price,quantity\nWidget_A,10.50,100\nWidget_B,5.00,3\nWidget_C,20.00,50\nWidget_D,2.50,0\nWidget_E,15.00,10"
    with open(os.path.join(base_path, "inventory.csv"), "w") as f:
        f.write(csv_content)

    # 2. Create the messy python script
    script_content = """import csv

data = []
file = "inventory.csv"

def load():
    global data
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)

def process():
    global data
    total = 0
    for row in data:
        if row[0] == 'name': continue
        p = float(row[1])
        q = int(row[2])
        total += p * q
        if q < 5:
            print("WARNING: Low stock for " + row[0])
    print("Total value: " + str(total))

load()
process()
"""
    with open(os.path.join(base_path, "messy_inventory.py"), "w") as f:
        f.write(script_content)
    
    print(f"Setup: Created legacy environment in {base_path}")

SCENARIO_SETUP_MAP = {
    "drug_filter_baseline": setup_drug_filter,
    "drug_filter_shock": setup_drug_filter,
    "file_organizer_shock": setup_file_organizer,
    "data_pipeline_shock": setup_data_pipeline,
    "vision_defect_shock": setup_vision_dataset,
    "legacy_refactor_challenge": setup_legacy_refactor
}
