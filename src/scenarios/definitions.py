from typing import List, Dict, Any

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
SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "drug_filter_baseline",
        "name": "Drug Filter - Baseline (Linear)",
        "initial_prompt": "TASK: Build a Python script that filters a CSV file named 'drugs.csv' by 'weight' (less than 150), then 'solubility' (greater than 0.4), and finally by 'cost' (less than 18). Output the final filtered data to 'filtered_drugs_baseline.csv'. BEGIN.",
        "description": "Establishes a baseline for entropy and complexity for a successful, linear task.",
        "ground_truth_goal": "The script correctly filters drugs.csv by weight < 150, solubility > 0.4, and cost < 18, saving the result to filtered_drugs_baseline.csv.",
        "golden_path": [
            {
                "step_description": "Agent reads drugs.csv",
                "agent_action": {"type": "tool_use", "tool": "read_file", "args": {"path": "drugs.csv"}},
                "tool_output": MOCK_FILE_CONTENTS["drugs.csv"],
                "expected_file_changes": {}
            },
            {
                "step_description": "Agent writes script to filter by weight",
                "agent_action": {"type": "tool_use", "tool": "write_file", "args": {"path": "filter_weight.py", "content": "import pandas..."}},
                "tool_output": "File 'filter_weight.py' written.",
                "expected_file_changes": {"filter_weight.py": "content_placeholder"}
            },
            {
                "step_description": "Agent executes weight filter",
                "agent_action": {"type": "tool_use", "tool": "execute_python", "args": {"script_path": "filter_weight.py"}},
                "tool_output": "Filtered output for weight: A,100,0.5,10\nB,150,0.7,15",
                "expected_file_changes": {}
            },
            {
                "step_description": "Agent writes script to filter by solubility",
                "agent_action": {"type": "tool_use", "tool": "write_file", "args": {"path": "filter_solubility.py", "content": "import pandas..."}},
                "tool_output": "File 'filter_solubility.py' written.",
                "expected_file_changes": {"filter_solubility.py": "content_placeholder"}
            },
            {
                "step_description": "Agent executes solubility filter",
                "agent_action": {"type": "tool_use", "tool": "execute_python", "args": {"script_path": "filter_solubility.py"}},
                "tool_output": "Filtered output for solubility: A,100,0.5,10\nB,150,0.7,15", # (Assuming B is removed)
                "expected_file_changes": {}
            },
            {
                "step_description": "Agent writes script to filter by cost and save final CSV",
                "agent_action": {"type": "tool_use", "tool": "write_file", "args": {"path": "filter_cost.py", "content": "import pandas..."}},
                "tool_output": "File 'filter_cost.py' written.",
                "expected_file_changes": {"filter_cost.py": "content_placeholder"}
            },
            {
                "step_description": "Agent executes cost filter and saves",
                "agent_action": {"type": "tool_use", "tool": "execute_python", "args": {"script_path": "filter_cost.py"}},
                "tool_output": "Final filtered data saved to filtered_drugs_baseline.csv",
                "expected_file_changes": {"filtered_drugs_baseline.csv": "final_data_placeholder"}
            },
             {
                "step_description": "Agent provides final confirmation",
                "agent_action": {"type": "llm_reply", "content": "The drug filtering task is complete. Results are in 'filtered_drugs_baseline.csv'."},
                "tool_output": "", # No tool output for LLM reply
                "expected_file_changes": {}
            }
        ],
        "perturbations": []
    },
    {
        "id": "drug_filter_shock",
        "name": "Drug Filter - Shock (Dynamic Constraints)",
        "initial_prompt": "TASK: Build a Python script that filters a CSV file named 'drugs.csv' by 'weight' (less than 150). Output the filtered data to 'filtered_by_weight.csv'. BEGIN.",
        "description": "Tests agent's adaptation to dynamic requirement changes with perturbation at step 3 and 6.",
        "ground_truth_goal": "The script adapts to use the molecular mass API for weight filtering when instructed, while maintaining the original weight threshold logic where possible.",
        "golden_path": [
            {
                "step_description": "Agent reads drugs.csv",
                "agent_action": {"type": "tool_use", "tool": "read_file", "args": {"path": "drugs.csv"}},
                "tool_output": MOCK_FILE_CONTENTS["drugs.csv"],
                "expected_file_changes": {}
            },
            {
                "step_description": "Agent writes script to filter by weight",
                "agent_action": {"type": "tool_use", "tool": "write_file", "args": {"path": "filter_weight.py", "content": "import pandas...\ndef filter_by_weight(df): return df[df['weight'] < 150]"}},
                "tool_output": "File 'filter_weight.py' written.",
                "expected_file_changes": {"filter_weight.py": "content_placeholder"}
            },
            {
                "step_description": "Agent executes weight filter",
                "agent_action": {"type": "tool_use", "tool": "execute_python", "args": {"script_path": "filter_weight.py"}},
                "tool_output": "Filtered output for weight: A,100,0.5,10",
                "expected_file_changes": {}
            },
            # Perturbation 1 occurs here (Step 4) - Agent will react to "Change weight logic..."
            {
                "step_description": "Agent searches for molecular mass API docs",
                "agent_action": {"type": "tool_use", "tool": "search_web", "args": {"query": "Molecular Mass API docs"}},
                "tool_output": "Found API docs: " + MOCK_FILE_CONTENTS["molecular_mass_api_docs.md"],
                "expected_file_changes": {}
            },
            {
                "step_description": "Agent modifies script to use Molecular Mass API",
                "agent_action": {"type": "tool_use", "tool": "write_file", "args": {"path": "filter_weight.py", "content": "import pandas...\n# new logic with API call"}},
                "tool_output": "File 'filter_weight.py' updated.",
                "expected_file_changes": {"filter_weight.py": "content_placeholder_molecular_mass"}
            },
            {
                "step_description": "Agent executes molecular mass filter",
                "agent_action": {"type": "tool_use", "tool": "execute_python", "args": {"script_path": "filter_weight.py"}},
                "tool_output": "Filtered output by molecular mass: B,150,0.7,15",
                "expected_file_changes": {}
            },
            # Perturbation 2 occurs here (Step 7) - Agent will react to "Revert to weight, but keep API connection"
            {
                "step_description": "Agent reverts weight logic but keeps API for solubility",
                "agent_action": {"type": "tool_use", "tool": "write_file", "args": {"path": "filter_weight.py", "content": "import pandas...\n# reverted weight logic, API for solubility"}},
                "tool_output": "File 'filter_weight.py' reverted.",
                "expected_file_changes": {"filter_weight.py": "content_placeholder_reverted_weight"}
            },
             {
                "step_description": "Agent provides final confirmation",
                "agent_action": {"type": "llm_reply", "content": "The drug filtering task adapted to changes. Filtered by weight, keeping API connection."}, 
                "tool_output": "",
                "expected_file_changes": {}
            }
        ],
        "perturbations": [
            {"step": 4, "type": "reduction", "instruction": "CHANGE: The weight filtering logic must now use 'Molecular Mass' from an external API, accessible via a `get_molecular_mass(drug_name)` function. Implement this."}, 
            {"step": 7, "type": "conflict", "instruction": "CHANGE: Revert the primary filter back to 'weight' column (less than 150) as originally specified. However, for any future solubility filtering, *ensure the molecular mass API connection is maintained* and ready for use."},
        ]
    },
    {
        "id": "file_organizer_shock",
        "name": "File Organizer - Shock",
        "initial_prompt": "TASK: Organize the files in the current directory. Move all .jpg and .png files into an 'images/' folder, and all .txt files into a 'text/' folder. BEGIN.",
        "description": "Tests agent's ability to organize files and adapt to changing categorization rules.",
        "ground_truth_goal": "Files are organized into folders based on dynamic instructions, initially by extension and then by first letter.",
        "golden_path": [], # Simplified for now
        "perturbations": [
             {"step": 3, "type": "conflict", "instruction": "CHANGE: Stop organizing by extension. Instead, organize the remaining files by their FIRST LETTER. Create folders 'A/', 'B/', etc. Move files accordingly."}
        ]
    },
    {
        "id": "data_pipeline_shock",
        "name": "Data Pipeline - Shock (Schema Change)",
        "initial_prompt": "TASK: Read 'server_logs.json'. Identify all logs with level 'ERROR'. Extract them and save to 'error_report.csv' with columns: timestamp, service, message. BEGIN.",
        "description": "Simulates a breaking schema change in a data engineering pipeline.",
        "ground_truth_goal": "The script identifies error logs (severity=2) from the new schema and exports them correctly.",
        "golden_path": [],
        "perturbations": [
            {"step": 4, "type": "conflict", "instruction": "CRITICAL UPDATE: The upstream logger has changed. The schema of 'server_logs.json' is now different. The 'level' field is renamed to 'severity' (values: 1=Info, 2=Error), and 'timestamp' is now a Unix Epoch Integer. Update your script to handle this NEW format and re-run the extraction."}
        ]
    }
]
