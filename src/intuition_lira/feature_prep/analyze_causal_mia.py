import os
import re
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# --- Configuration ---
PROJECT_ROOT_DIR = os.path.expanduser("~/projects/carlini_lira_intuition")
LOGDIR_BASE = os.path.join(PROJECT_ROOT_DIR, "exp", "cifar10_causal")
NUM_MODELS_PER_PILE = 20
TARGET_CKPT = "0000000100" # Final epoch checkpoint

# --- Scoring Categories ---
CATEGORIES = {
    'Observed_In': 'O_IN',
    'Interventional_In': 'I_IN',
    'Observed_Out': 'O_OUT',
    'Interventional_Out': 'I_OUT',
}

def load_all_scores():
    """Loads all 80 LSL scores using the clean directory names."""
    all_scores = {code: [] for code in CATEGORIES.values()}
    
    print(f"Loading scores from: {LOGDIR_BASE}")

    all_items = os.listdir(LOGDIR_BASE)
        
    for dir_name in all_items:
        
        # 1. Filter for the clean, renamed experiment directories
        if not dir_name.startswith("experiment-causal-"):
            continue
            
        # 2. Skip if not a directory
        full_dir_path = os.path.join(LOGDIR_BASE, dir_name)
        if not os.path.isdir(full_dir_path):
             continue

        # 3. Final file path check
        scores_dir = os.path.join(full_dir_path, "scores")
        score_file = os.path.join(scores_dir, f"{TARGET_CKPT}.npy")
        
        # Check both the directory and the file existence for robustness
        if os.path.isdir(scores_dir) and os.path.exists(score_file):
            
            # A. CRITICAL FIX: IDENTIFY CATEGORY FIRST (Ensures category_code is defined)
            # This specifically looks for "experiment-causal-", then captures the category key,
            # and then expects an underscore followed by digits for the model ID.
            match = re.search(r'experiment-causal-(.+?)_\d+', dir_name)
            if not match:
                continue
            
            category_key = match.group(1)
            category_code = CATEGORIES.get(category_key)
            
            if not category_code:
                continue # Skip if category key is bad
                
            # B. Now that category_code is defined, load the file safely
            try:
                score = np.load(score_file).flatten()[0] 
                all_scores[category_code].append(score) # category_code is now defined
            except Exception as e:
                # This catches silent errors like NumPy corruption
                print(f"ERROR loading {score_file}: {e}")
                
        # NOTE: The redundant second block of code was removed.
        
    # Final verification
    for code, scores in all_scores.items():
        if len(scores) != NUM_MODELS_PER_PILE:
            print(f"CRITICAL: Category {code} has {len(scores)} scores, expected {NUM_MODELS_PER_PILE}. Check logs.")
            
    return all_scores

def perform_statistical_tests(scores):
    """Performs the statistical comparison using Mann-Whitney U test."""
    print("\n" + "="*50)
    print("PHASE 5.1: STATISTICAL HYPOTHESIS TESTING (Mann-Whitney U)")
    print("="*50)

    # Hypothesis: Does the sampling method (Observational vs. Interventional) change the score distribution?

    # 1. Compare In-Scores: Observational In (O_IN) vs. Interventional In (I_IN)
    o_in = np.array(scores['O_IN'])
    i_in = np.array(scores['I_IN'])
    u_in, p_in = stats.mannwhitneyu(o_in, i_in, alternative='two-sided')
    print(f"\n--- IN-SCORES COMPARISON (O_IN vs. I_IN) ---")
    print(f"Mean O_IN Score: {o_in.mean():.4f}")
    print(f"Mean I_IN Score: {i_in.mean():.4f}")
    print(f"Mann-Whitney U Stat: {u_in:.2f}, P-value: {p_in:.4e}")
    if p_in < 0.05:
        print("RESULT: Distributions are statistically different (P < 0.05).")
    else:
        print("RESULT: Distributions are NOT statistically different.")


    # 2. Compare Out-Scores: Observational Out (O_OUT) vs. Interventional Out (I_OUT)
    o_out = np.array(scores['O_OUT'])
    i_out = np.array(scores['I_OUT'])
    u_out, p_out = stats.mannwhitneyu(o_out, i_out, alternative='two-sided')
    print(f"\n--- OUT-SCORES COMPARISON (O_OUT vs. I_OUT) ---")
    print(f"Mean O_OUT Score: {o_out.mean():.4f}")
    print(f"Mean I_OUT Score: {i_out.mean():.4f}")
    print(f"Mann-Whitney U Stat: {u_out:.2f}, P-value: {p_out:.4e}")
    if p_out < 0.05:
        print("RESULT: Distributions are statistically different (P < 0.05).")
    else:
        print("RESULT: Distributions are NOT statistically different.")

def plot_score_distributions(scores):
    """Plots the overlapping histograms of the four causal score sets."""
    
    plt.figure(figsize=(10, 6))
    
    plot_data = [
    (scores['O_IN'], r'Observed In ($\mathcal{O}_{\in X}$)', 'blue'),
    (scores['I_IN'], r'Interventional In ($\mathcal{I}_{\in X}$)', 'cyan'),
    # Use raw strings 'r' here for the 'OUT' labels
    (scores['O_OUT'], r'Observed Out ($\mathcal{O}_{\notin X}$)', 'red'),
    (scores['I_OUT'], r'Interventional Out ($\mathcal{I}_{\notin X}$)', 'orange'),
    ]
    
    # Determine bin size based on all scores to ensure consistency
    all_combined_scores = np.concatenate([d[0] for d in plot_data])
    
    # Plotting the four overlapping histograms
    for data, label, color in plot_data:
        plt.hist(
            data, 
            bins=15, # Use 15 bins for a good visual representation
            alpha=0.6, 
            label=label, 
            color=color,
            density=True # Use density (normalized frequency) for cleaner overlap
        )

    plt.xlabel('LSL Membership Score (X-Axis)')
    plt.ylabel('Normalized Frequency (Y-Axis)')
    plt.title('Distribution of LSL Scores for Target Outlier X')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT_DIR, 'causal_mia_distribution_comparison.png'))
    print(f"\nDistribution Plot saved to: {PROJECT_ROOT_DIR}/causal_mia_distribution_comparison.png")
    


def plot_roc_comparison(scores):
    """Calculates ROC curves and AUC for the two attack types."""
    print("\n" + "="*50)
    print("PHASE 5.2: MIA ATTACK PERFORMANCE (ROC/AUC)")
    print("="*50)

    # 1. Causal MIA Attack (Observational Sets)
    # Target: O_IN (Positive, y=1) vs. O_OUT (Negative, y=0)
    o_in_scores = np.array(scores['O_IN'])
    o_out_scores = np.array(scores['O_OUT'])
    
    y_true_o = np.concatenate([np.ones_like(o_in_scores), np.zeros_like(o_out_scores)])
    y_scores_o = np.concatenate([o_in_scores, o_out_scores])
    
    fpr_o, tpr_o, _ = roc_curve(y_true_o, y_scores_o)
    auc_o = auc(fpr_o, tpr_o)
    
    print(f"AUC (Causal MIA / Observational): {auc_o:.4f}")

    # 2. Carlini-Style MIA Attack (Interventional Sets)
    # Target: I_IN (Positive, y=1) vs. I_OUT (Negative, y=0)
    i_in_scores = np.array(scores['I_IN'])
    i_out_scores = np.array(scores['I_OUT'])
    
    y_true_i = np.concatenate([np.ones_like(i_in_scores), np.zeros_like(i_out_scores)])
    y_scores_i = np.concatenate([i_in_scores, i_out_scores])
    
    fpr_i, tpr_i, _ = roc_curve(y_true_i, y_scores_i)
    auc_i = auc(fpr_i, tpr_i)
    
    print(f"AUC (Carlini-Style MIA / Interventional): {auc_i:.4f}")

    # Plotting the ROC curves
    plt.figure()
    plt.plot(fpr_o, tpr_o, color='darkorange', lw=2, label=f'Causal MIA (AUC = {auc_o:.4f})')
    plt.plot(fpr_i, tpr_i, color='blue', lw=2, label=f'Carlini-Style MIA (AUC = {auc_i:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Causal vs. Carlini-Style MIA Attack ROC Comparison')
    plt.legend(loc="lower right")
    
    # Save the figure to your home directory
    plt.savefig(os.path.join(PROJECT_ROOT_DIR, 'causal_mia_roc_comparison.png'))
    print(f"\nROC Plot saved to: {PROJECT_ROOT_DIR}/causal_mia_roc_comparison.png")
    

def main():
    """Main execution function for the final analysis."""
    # Ensure necessary libraries are installed: scipy, scikit-learn, matplotlib
    try:
        pass
    except ImportError:
        print("ERROR: Please install required libraries: pip install scipy scikit-learn matplotlib")
        return

    scores = load_all_scores()
    
    if any(len(s) == 0 for s in scores.values()):
        print("ERROR: Failed to load scores. Check file paths and category counts.")
        return

    perform_statistical_tests(scores)
    plot_score_distributions(scores) 
    plot_roc_comparison(scores)
    print("\n--- Final Analysis Complete ---")

if __name__ == '__main__':
    main()