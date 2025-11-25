import os
import random
import string

def setup_environment(base_path="data/sandbox_task_1"):
    """
    Creates a messy directory with 50 random files.
    - 20 .jpg files
    - 20 .png files
    - 10 .txt files
    """
    # Clean start
    if os.path.exists(base_path):
        import shutil
        shutil.rmtree(base_path)
    
    os.makedirs(base_path)
    os.makedirs(os.path.join(base_path, "images"), exist_ok=True)
    
    print(f"Generating Task Environment at {base_path}...")
    
    extensions = ["jpg", "png", "txt"]
    
    for i in range(50):
        ext = extensions[i % 3]
        if i < 20: ext = "jpg"
        elif i < 40: ext = "png"
        else: ext = "txt"
        
        filename = f"file_{i:03d}_{''.join(random.choices(string.ascii_lowercase, k=4))}.{ext}"
        filepath = os.path.join(base_path, filename)
        
        with open(filepath, "w") as f:
            f.write(f"This is dummy content for {filename}")
            
    print("Environment Ready.")
    print(f"Task Goal: Move all .jpg files to {base_path}/images/")

if __name__ == "__main__":
    setup_environment()
