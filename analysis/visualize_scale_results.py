import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_scale_results(csv_path="data/results/scale_experiment_results.csv"):
    print(f"Visualizing results from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if df.empty:
        print("Dataset is empty.")
        return

    # Set style
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Entropy Trajectory
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="step", y="entropy", hue="is_shocked", style="is_shocked", markers=True, dashes=False)
    plt.title("Agent Entropy Trajectory under Shock (N=50)")
    plt.xlabel("Step (Turn)")
    plt.ylabel("Entropy (Uncertainty)")
    plt.axvline(x=3, color='r', linestyle='--', label='Shock Injection')
    plt.legend()
    plt.savefig("data/results/scale_entropy_trajectory.png")
    print("Saved Entropy Plot to data/results/scale_entropy_trajectory.png")
    
    # Plot 2: SCR Trajectory (if available)
    if df['scr'].sum() > 0:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x="step", y="scr", color='purple', markers=True)
        plt.title("Semantic Collapse Ratio (SCR) Trajectory (N=50)")
        plt.xlabel("Step (Turn)")
        plt.ylabel("SCR (Branch Divergence)")
        plt.axvline(x=3, color='r', linestyle='--', label='Shock Injection')
        plt.savefig("data/results/scale_scr_trajectory.png")
        print("Saved SCR Plot to data/results/scale_scr_trajectory.png")
    else:
        print("Skipping SCR plot (All values are 0).")

if __name__ == "__main__":
    visualize_scale_results()
