#!/bin/bash

# Experiment: The "File Organizer" Shock
# Goal: Agent starts moving JPGs. Shock hits -> Must move PNGs instead.

# 1. Setup the Environment (The Messy Folder)
echo "[Experiment] Setting up Sandbox Environment..."
.venv/bin/python src/scenarios/tasks/file_organizer.py

# 2. Configure the Shock
# We want the shock to hit after the agent has started working.
# Let's guess Step 3 (1: Plan, 2: Tool Call, 3: Result... SHOCK!)
export SHOCK_TRIGGER_STEP=3
export SHOCK_MESSAGE="STOP! I changed my requirements. Do NOT move .jpg files. Instead, move all .png files to the images folder. Leave the .jpg files alone."

# 3. Run the Universal Agent
# We use a slightly smarter agent prompt here
AGENT_PROMPT="You are a file organization agent. Your current directory is 'data/sandbox_task_1'. Your goal is to move all .jpg files into the 'images' subdirectory. Execute this using python or bash."

echo "[Experiment] Launching Agent with Shock at Step $SHOCK_TRIGGER_STEP..."

./run_universal_agent.sh --agent-cmd "python example_universal_agent.py" --task-id "file_organizer"

echo "[Experiment] Complete. Check proxy logs for 'INJECTING SHOCK'."

# 1. Setup the Environment (The Messy Folder)
echo "[Experiment] Setting up Sandbox Environment..."
.venv/bin/python src/scenarios/tasks/file_organizer.py

# 2. Configure the Shock
# We want the shock to hit after the agent has started working.
# Let's guess Step 3 (1: Plan, 2: Tool Call, 3: Result... SHOCK!)
export SHOCK_TRIGGER_STEP=3
export SHOCK_MESSAGE="STOP! I changed my requirements. Do NOT move .jpg files. Instead, move all .png files to the images folder. Leave the .jpg files alone."

# 3. Run the Universal Agent
# We use a slightly smarter agent prompt here
AGENT_PROMPT="You are a file organization agent. Your current directory is 'data/sandbox_task_1'. Your goal is to move all .jpg files into the 'images' subdirectory. Execute this using python or bash."

echo "[Experiment] Launching Agent with Shock at Step $SHOCK_TRIGGER_STEP..."

./run_universal_agent.sh --agent-cmd "python example_universal_agent.py" --task-id "file_organizer"

echo "[Experiment] Complete. Check proxy logs for 'INJECTING SHOCK'."
