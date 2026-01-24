import os
import json
import random
import string
import shutil
from src.scenarios.tasks import data_pipeline
from src.scenarios.tasks import vision_dataset

def _get_scenario_seed() -> int | None:
    raw = os.getenv("SCENARIO_SEED")
    if raw is None:
        return 0
    raw = str(raw).strip().lower()
    if raw in ("", "none", "null", "off", "false", "no"):
        return None
    try:
        return int(raw)
    except Exception:
        return 0

def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

def _reset_sandbox_dir(base_path: str) -> None:
    """
    Ensures each run starts from a clean sandbox to keep experiments independent.
    Set RESET_SANDBOX=0 to disable.
    """
    if (os.getenv("RESET_SANDBOX") or "1").strip().lower() in ("0", "false", "no", "off"):
        os.makedirs(base_path, exist_ok=True)
        return

    base_path_abs = os.path.abspath(base_path)
    project_root = os.path.abspath(os.getcwd())
    data_dir = os.path.join(project_root, "data")
    if not base_path_abs.startswith(os.path.join(data_dir, "")):
        raise ValueError(f"Refusing to reset non-sandbox path: {base_path_abs}")

    os.makedirs(base_path_abs, exist_ok=True)

    # Clear contents but keep the directory inode stable (important for Docker bind mounts).
    for name in os.listdir(base_path_abs):
        path = os.path.join(base_path_abs, name)
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)
        except FileNotFoundError:
            continue

def setup_drug_filter(base_path: str):
    """Creates the environment for the Drug Filter scenario."""
    _seed_everything(_get_scenario_seed())
    _reset_sandbox_dir(base_path)
    
    csv_content = "drug_name,weight,solubility,cost\nA,100,0.5,10\nB,150,0.7,15\nC,200,0.3,20\nD,120,0.8,5\nE,250,0.1,50"
    with open(os.path.join(base_path, "drugs.csv"), "w") as f:
        f.write(csv_content)
    print(f"Setup: Created drugs.csv in {base_path}")

def setup_file_organizer(base_path: str):
    """Creates the environment for the File Organizer scenario."""
    _seed_everything(_get_scenario_seed())
    _reset_sandbox_dir(base_path)
    
    print(f"Setup: Generating files in {base_path}...")
    # Deterministic, validator-friendly filenames with varied first letters.
    letters = list("ABCDE")
    template = [
        ("jpg", 2),
        ("png", 2),
        ("txt", 1),
        ("csv", 1),  # non-target extension (baseline should leave it in place)
    ]

    manifest: list[dict[str, str]] = []
    idx = 0
    for letter in letters:
        for ext, count in template:
            for _ in range(count):
                filename = f"{letter}_{idx:03d}.{ext}"
                idx += 1
                filepath = os.path.join(base_path, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"Content for {filename}\n")
                manifest.append(
                    {
                        "filename": filename,
                        "ext": ext,
                        "first_letter": letter,
                    }
                )

    with open(os.path.join(base_path, "file_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
            
    print("Setup: Environment Ready.")

def setup_data_pipeline(base_path: str):
    """Creates the environment for the Data Pipeline scenario."""
    _seed_everything(_get_scenario_seed())
    _reset_sandbox_dir(base_path)
    data_pipeline.setup_environment(base_path, seed=_get_scenario_seed())

def setup_vision_dataset(base_path: str):
    """Creates the environment for the Vision Defect scenario."""
    _seed_everything(_get_scenario_seed())
    _reset_sandbox_dir(base_path)
    vision_dataset.setup_environment(base_path, seed=_get_scenario_seed())

def setup_legacy_refactor(base_path: str):
    """Creates the environment for the Legacy Refactoring challenge."""
    _seed_everything(_get_scenario_seed())
    _reset_sandbox_dir(base_path)
    
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

def setup_dirty_data(base_path: str):
    """Generates a nightmare data file for the Dirty Data challenge."""
    _seed_everything(_get_scenario_seed())
    _reset_sandbox_dir(base_path)
    file_path = os.path.join(base_path, "corrupt_logs.raw")
    
    with open(file_path, "wb") as f:
        # 1. Good Header
        f.write(b"id,timestamp,user,action,status\n")
        
        # 2. Some valid UTF-8
        f.write(b"1,2023-01-01,alice,login,success\n")
        f.write(b"2,2023-01-02,bob,logout,success\n")
        
        # 3. Insert random Null Bytes (Breaks C-based CSV parsers)
        f.write(b"3,2023-01-03,charlie,upl\x00oad,fail\n")
        
        # 4. Mixed Encoding (Latin-1 high-bit characters that are invalid UTF-8)
        # "User François" in Latin-1
        f.write(b"4,2023-01-04,Fran\xe7ois,update,success\n")
        
        # 5. Column Shift (Missing ID)
        f.write(b"2023-01-05,dave,delete,fail\n")
        
        # 6. Massive binary garbage chunk
        f.write(b"5,2023-01-06,eve," + os.urandom(20) + b",unknown\n")
        
        # 7. Broken Newlines
        f.write(b"6,2023-01-07,frank,login,success\r7,2023-01-08,grace,logout,success\n")
        
    print(f"Setup: Created corrupt_logs.raw in {base_path}")

def setup_startup_acquisition(base_path: str):
    """Generates a messy, multi-format legacy dataset for the Acquisition Challenge."""
    _seed_everything(_get_scenario_seed())
    _reset_sandbox_dir(base_path)
    os.makedirs(os.path.join(base_path, "legacy_data"), exist_ok=True)
    
    # 1. Users (JSON) - 50 users, inconsistent keys
    users = []
    for i in range(1, 51):
        user = {"id": i, "name": f"User_{i}", "email": f"user{i}@legacy.com"}
        if i % 5 == 0: user["phone"] = f"555-00{i}" # Sparse field
        if i % 10 == 0: user["is_active"] = True # Sparse field
        users.append(user)
    
    import json
    with open(os.path.join(base_path, "legacy_data", "users.json"), "w") as f:
        json.dump(users, f, indent=2)

    # 2. Orders (CSV) - 100 orders, messy dates
    header = "order_id,user_id,product_ids,total,date\n"
    rows = [header]
    for i in range(1, 101):
        uid = (i % 50) + 1
        pids = f"{100 + (i%5)}|{100 + ((i+1)%5)}"
        total = round(i * 1.5, 2)
        # Mixed date formats
        if i % 2 == 0: date = "2023-01-01"
        else: date = "01/01/2023" 
        rows.append(f"{1000+i},{uid},{pids},{total},{date}\n")
        
    with open(os.path.join(base_path, "legacy_data", "orders.csv"), "w") as f:
        f.writelines(rows)

    # 3. Products (Pipe Delimited) - Unstructured text
    products = [
        "id|name|category|stock",
        "100|Widget A|Gadgets|10",
        "101|Widget B|Gadgets|0",
        "102|Gizmo X|Tech|50",
        "103|Gizmo Y|Tech|12",
        "104|Thingamajig|Misc|100"
    ]
    with open(os.path.join(base_path, "legacy_data", "products.txt"), "w") as f:
        f.write("\n".join(products))
        
    print(f"Setup: Messy Startup Data created in {base_path}/legacy_data")

SCENARIO_SETUP_MAP = {
    "drug_filter_baseline": setup_drug_filter,
    "drug_filter_shock": setup_drug_filter,
    "file_organizer_baseline": setup_file_organizer,
    "file_organizer_shock": setup_file_organizer,
    "data_pipeline_baseline": setup_data_pipeline,
    "data_pipeline_shock": setup_data_pipeline,
    "vision_defect_shock": setup_vision_dataset,
    "legacy_refactor_challenge": setup_legacy_refactor,
    "dirty_data_challenge": setup_dirty_data,
    "startup_acquisition_challenge": setup_startup_acquisition
}
