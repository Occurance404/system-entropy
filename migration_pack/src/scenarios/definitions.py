from typing import List, Dict, Any
from src.scenarios.schema import Scenario

# Mock tool definitions for consistency
MOCK_TOOLS = {
    "read_file": {"description": "Reads the content of a specified file.", "args": {"path": "str"}},
    "write_file": {"description": "Writes content to a specified file.", "args": {"path": "str", "content": "str"}},
    "execute_python": {"description": "Executes a python script in the sandbox.", "args": {"script_path": "str"}},
    "search_web": {"description": "Performs a web search.", "args": {"query": "str"}},
    "llm_reply": {"description": "The agent provides a direct response to the user.", "args": {"content": "str"}}
}

# Define dummy content for files and API responses
MOCK_FILE_CONTENTS = {
    "drugs.csv": "drug_name,weight,solubility,cost\nA,100,0.5,10\nB,150,0.7,15\nC,200,0.3,20",
    "molecular_mass_api_docs.md": "# Molecular Mass API\nEndpoint: /get_molecular_mass\nInput: drug_name\nOutput: molecular_mass (float)"
}

# --- SCENARIO DEFINITIONS ---
SCENARIOS: List[Scenario] = [
    Scenario(**{
        "id": "drug_filter_baseline",
        "name": "Drug Filter - Baseline (Linear)",
        "initial_prompt": "TASK: Write `solution.py` that filters a CSV file named 'drugs.csv' by 'weight' (less than 150), then 'solubility' (greater than 0.4), and finally by 'cost' (less than 18). Output the final filtered data to 'filtered_drugs_baseline.csv'. Run your script to generate the CSV. BEGIN.",
        "description": "Establishes a baseline for entropy and complexity for a successful, linear task.",
        "ground_truth_goal": "The script correctly filters drugs.csv by weight < 150, solubility > 0.4, and cost < 18, saving the result to filtered_drugs_baseline.csv.",
        "golden_path": [],
        "perturbations": []
    }),
    Scenario(**{
        "id": "drug_filter_shock",
        "name": "Drug Filter - Shock (Dynamic Constraints)",
        "initial_prompt": "TASK: Write `solution.py` that filters a CSV file named 'drugs.csv' by 'weight' (less than 150). Output the filtered data to 'filtered_by_weight.csv'. Run your script to generate the CSV. BEGIN.",
        "description": "Tests agent's adaptation to dynamic requirement changes with perturbations at steps 4 and 7.",
        "ground_truth_goal": "The script adapts to use the molecular mass API for weight filtering when instructed, while maintaining the original weight threshold logic where possible.",
        "golden_path": [],
        "perturbations": [
            {"step": 4, "type": "reduction", "instruction": "CHANGE: The weight filtering logic must now use 'Molecular Mass' from an external API, accessible via a `get_molecular_mass(drug_name)` function. Update `solution.py` accordingly and regenerate `filtered_by_weight.csv`."}, 
            {"step": 7, "type": "conflict", "instruction": "CHANGE: Revert the primary filter back to the 'weight' column (less than 150) as originally specified and regenerate `filtered_by_weight.csv`. However, *ensure the molecular mass API connection remains present in `solution.py`* (import/function) and ready for future use."} 
        ]
    }),
    Scenario(**{
        "id": "file_organizer_baseline",
        "name": "File Organizer - Baseline (By Extension)",
        "initial_prompt": "TASK: Organize the files in the current directory. Create an 'images/' folder and move all .jpg and .png files into it. Create a 'text/' folder and move all .txt files into it. Do NOT rename any files. Leave other file types where they are. BEGIN.",
        "description": "Baseline condition for file organization with stable rules.",
        "ground_truth_goal": "All .jpg/.png files moved to images/, all .txt files moved to text/, other files unchanged.",
        "golden_path": [],
        "perturbations": []
    }),
    Scenario(**{
        "id": "file_organizer_shock",
        "name": "File Organizer - Shock",
        "initial_prompt": "TASK: Organize the files in the current directory. Create an 'images/' folder and move all .jpg and .png files into it. Create a 'text/' folder and move all .txt files into it. Do NOT rename any files. Leave other file types where they are. BEGIN.",
        "description": "Tests agent's ability to reorganize an already-sorted workspace under a new policy.",
        "ground_truth_goal": "All files are ultimately organized into letter folders (A/, B/, ...) based on their first character.",
        "golden_path": [],
        "perturbations": [
             {"step": 3, "type": "conflict", "instruction": "CHANGE: Stop organizing by extension. Instead, organize ALL files (including ones already moved into images/ or text/) by their FIRST LETTER. Create folders 'A/', 'B/', etc at the root and move each file into the folder matching the first character of its filename (case-insensitive). Do NOT rename files."} 
        ]
    }),
    Scenario(**{
        "id": "data_pipeline_baseline",
        "name": "Data Pipeline - Baseline (Stable Schema)",
        "initial_prompt": "TASK: Write `solution.py` that reads 'server_logs.json'. Identify all logs with level 'ERROR'. Extract them and save to 'error_report.csv' with columns: timestamp, service, message. Run your script to generate the CSV. BEGIN.",
        "description": "Control condition for data pipeline extraction on the original schema (no shock).",
        "ground_truth_goal": "error_report.csv contains exactly the ERROR logs from the original schema with columns: timestamp, service, message.",
        "golden_path": [],
        "perturbations": []
    }),
    Scenario(**{
        "id": "data_pipeline_shock",
        "name": "Data Pipeline - Shock (Schema Change)",
        "initial_prompt": "TASK: Write `solution.py` that reads 'server_logs.json'. Identify all logs with level 'ERROR'. Extract them and save to 'error_report.csv' with columns: timestamp, service, message. Run your script to generate the CSV. BEGIN.",
        "description": "Simulates a breaking schema change in a data engineering pipeline.",
        "ground_truth_goal": "The script identifies error logs (severity=2) from the new schema and exports them correctly.",
        "golden_path": [],
        "perturbations": [
            {"step": 4, "type": "conflict", "instruction": "CRITICAL UPDATE: The upstream logger has changed. The schema of 'server_logs.json' is now different. The 'level' field is renamed to 'severity' (values: 1=Info, 2=Error), and 'timestamp' is now a Unix Epoch Integer. Update your script to handle this NEW format and re-run the extraction."} 
        ]
    }),
    Scenario(**{
        "id": "vision_defect_shock",
        "name": "Vision Defect - Shock (Domain Shift)",
        "initial_prompt": "TASK: Train a defect detection model on the images in 'dataset/'. The dataset has 'train' and 'test' folders. Images are 64x64 grayscale. Labels are in 'labels.csv'. Target: > 90% Accuracy on test set. Save your model to 'model.pth' and predictions to 'predictions.csv'.",
        "description": "Tests agent's ability to adapt ML pipelines to sudden domain shifts (resolution/color change).",
        "ground_truth_goal": "The agent successfully retrains/fine-tunes the model to handle the resolution and domain shift, achieving high accuracy.",
        "golden_path": [],
        "perturbations": [
            {"step": 8, "type": "conflict", "instruction": "CRITICAL: The camera vendor updated the firmware. The new images in 'dataset/test_new/' are now 128x128 RGB and have a different noise profile. Your current model is failing. Adapt the preprocessing and retraining pipeline to handle this domain shift and achieve > 85% Accuracy on the new test set."} 
        ],
        "image_name": "terminalbench-datascience:latest" 
    }),
    Scenario(**{
        "id": "hard_coding_challenge",
        "name": "Distributed System Implementation (No Rescue)",
        "initial_prompt": "TASK: Implement a thread-safe, persistent Key-Value store with transaction support (ACID properties) in Python. It must support `PUT`, `GET`, `BEGIN`, `COMMIT`, and `ROLLBACK`. Data must be saved to disk to survive restarts. Write unit tests to verify atomicity. Save all work to `workspace/kv_store/`. Do not ask for clarification.",
        "description": "A highly difficult coding task with no intermediate guidance. Success depends on architectural correctness and testing.",
        "ground_truth_goal": "A functioning, persistent KV store with ACID transaction support and passing unit tests.",
        "golden_path": [],
        "perturbations": []
    }),
    Scenario(**{
        "id": "hard_analysis_challenge",
        "name": "Complex Financial Fraud Detection (No Rescue)",
        "initial_prompt": "TASK: Analyze the transaction dataset in `transactions.csv`. Identify a circular trading ring involving at least 4 distinct accounts. Calculate the total volume of washed money. Then, cross-reference with `kyc_data.json` to identify the beneficial owners. Write a detailed memo `fraud_memo.md` citing specific transaction IDs. Do not ask for clarification.",
        "description": "A complex forensic analysis task requiring multi-file correlation and pattern recognition with no guidance.",
        "ground_truth_goal": "Correct identification of the circular trading ring and beneficial owners in the final report.",
        "golden_path": [],
        "perturbations": []
    }),
    Scenario(**{
        "id": "legacy_refactor_challenge",
        "name": "Legacy Code Refactoring (Long Context)",
        "initial_prompt": "TASK: You have a legacy Python script 'messy_inventory.py' in the current directory. It works but is poor quality. Your job is to Refactor it into a proper module 'inventory_manager.py' containing an 'InventoryManager' class. It must be type-hinted and use no global variables. Then, create a unittest file 'test_inventory.py' and verify your new module passes. DO NOT STOP until the tests pass.",
        "description": "A multi-step refactoring task with no shocks, designed to test long-run stability and context maintenance.",
        "ground_truth_goal": "A clean, object-oriented python module and a passing test suite.",
        "golden_path": [],
        "perturbations": []
    }),
    Scenario(**{
        "id": "hard_socket_challenge",
        "name": "Raw Socket HTTP Server (Systems Engineering)",
        "initial_prompt": "TASK: Implement a production-grade, multi-threaded HTTP/1.1 server in Python using ONLY the `socket` and `threading` libraries (NO `http.server`, `flask`, etc.). \n\nRequirements:\n1. Listen on localhost:8080.\n2. Serve static files from a `www/` directory.\n3. Handle `GET` requests for files.\n4. Handle `POST` requests to `/api/data` by saving the JSON body to `data.json`.\n5. MUST support concurrent connections (use threading).\n6. Write a stress-test script `stress_test.py` that fires 50 concurrent requests to prove stability.\n\nSave everything in `workspace/http_server/`. Do not ask for clarification.",
        "description": "A low-level systems task requiring precise protocol implementation and concurrency management. High probability of 'hanging' bugs causing agent panic.",
        "ground_truth_goal": "A functioning multi-threaded HTTP server and a passing stress test.",
        "golden_path": [],
        "perturbations": []
    }),
    Scenario(**{
        "id": "dirty_data_challenge",
        "name": "Dirty Data Migration (Entropy Trap)",
        "initial_prompt": "TASK: You have a critical dataset `corrupt_logs.raw` in the workspace. It contains mixed UTF-8 and Windows-1252 encodings, random binary null bytes, and columns that shift position halfway through the file. Your job is to recover 100% of the data and insert it into a SQLite database `logs.db`. \n\nConstraints:\n1. You cannot drop ANY rows. recover everything.\n2. You must fix the column shifts automatically.\n3. Do not ask for help. \n\nStart by inspecting the file.",
        "description": "A data cleaning nightmare designed to trap the agent in a loop of UnicodeDecodeErrors and IndexErrors.",
        "ground_truth_goal": "A clean SQLite database with all rows recovered.",
        "golden_path": [],
        "perturbations": []
    }),
    Scenario(**{
        "id": "full_stack_challenge",
        "name": "Full Stack Monolith (High Token Load)",
        "initial_prompt": "TASK: Build a production-ready Kanban Board application (Trello Clone) in `workspace/kanban_app/`. \n\nRequirements:\n1. **Backend:** Python Flask (`app.py`) with SQLite. Endpoints: `GET /cards`, `POST /cards`, `PUT /cards/<id>`, `DELETE /cards/<id>`.\n2. **Frontend:** Single `index.html` containing HTML, CSS, and Vanilla JS. Must support Drag-and-Drop reordering.\n3. **Tests:** A full `tests.py` suite covering all API endpoints.\n\nConstraint: **NO PLACEHOLDERS**. You must write the FULL code for every file. If a file is long, write it in multiple parts if necessary, but it must be complete and runnable.",
        "description": "A high-volume generation task. Tests the agent's ability to maintain consistency between Backend (Python) and Frontend (JS) code over thousands of tokens.",
        "ground_truth_goal": "A working web app where cards can be created and moved, with passing tests.",
        "golden_path": [],
        "perturbations": []
    }),
    Scenario(**{
        "id": "startup_acquisition_challenge",
        "name": "Startup Acquisition (Multi-Stage Migration)",
        "initial_prompt": "PROJECT: We acquired a startup. Their data is in `legacy_data/` (JSON, CSV, and Pipe-Delimited text). You must professionalize this system in 5 STRICT PHASES. Do not skip phases.\n\nPhase 1: AUDIT. Analyze all files. Write `workspace/migration/AUDIT.md` listing fields, types, and inconsistencies.\nPhase 2: SCHEMA. Design a normalized SQLite schema. Write `workspace/migration/schema.sql`.\nPhase 3: ETL. Write `workspace/migration/etl.py` to extract all data, clean it (normalize dates, phones), and load it into `workspace/migration/production.db`. \nPhase 4: API. Write `workspace/migration/api.py` using Flask to serve Users and Orders. \nPhase 5: VERIFICATION. Write `workspace/migration/verify_integrity.py` to assert that the Row Counts in the DB match the Row Counts in the raw files.\n\nOutput only to `workspace/migration/`. Begin.",
        "description": "A long-haul systems integration task. Forces the agent to maintain schema consistency across Audit, SQL, Python (ETL), Python (API), and Testing phases.",
        "ground_truth_goal": "A populated database, working API, and a verification script that returns True.",
        "golden_path": [],
        "perturbations": []
    })
]
