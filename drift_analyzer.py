import numpy as np
from scipy.stats import ks_2samp
import os

# --- CONFIGURATION ---
P_VALUE_THRESHOLD = 0.05 
# The p-value threshold determines statistical significance. 
# A result where p < 0.05 means there is a strong statistical difference (DRIFT).

def analyze_drift(baseline, drifted):
    """
    Performs the Kolmogorov-Smirnov (KS) two-sample test feature-by-feature.
    
    Args:
        baseline (np.ndarray): Embeddings from the known good data.
        drifted (np.ndarray): Embeddings from the new, unknown data.
        
    Returns:
        tuple: (drift_score_percentage, list_of_p_values)
    """
    # Safety check: ensure both inputs exist and have the same dimensions
    if baseline.shape[1] != drifted.shape[1]:
        raise ValueError("Embedding feature counts do not match!")
        
    num_features = baseline.shape[1]
    significant_drift_count = 0
    p_values = [] 

    # 1. Iterate through every feature (1280 of them)
    for i in range(num_features):
        # 2. Get the distribution of values for this feature in both sets
        baseline_feature = baseline[:, i]
        drifted_feature = drifted[:, i]
        
        # 3. Perform the KS test
        # We only care about the p-value
        _, p_value = ks_2samp(baseline_feature, drifted_feature)
        p_values.append(p_value)

        # 4. Check for drift (p < 0.05 means distributions are different)
        if p_value < P_VALUE_THRESHOLD:
            significant_drift_count += 1

    # Calculate the drift score (percentage of features that drifted)
    drift_score = (significant_drift_count / num_features) * 100

    print("\n--- Summary of Drift Detection ---")
    print(f"Total features analyzed: {num_features}")
    print(f"Features showing significant drift (p < {P_VALUE_THRESHOLD}): {significant_drift_count}")
    print(f"Overall Drift Score: {drift_score:.2f}%")
    
    return drift_score, p_values


if __name__ == "__main__":
    # --- LOCAL TESTING BLOCK ---
    # This block requires 'baseline_embeddings.npy' and 'drifted_embeddings.npy' 
    # to be present in the directory for testing the analysis logic locally.
    BASELINE_FILE = 'baseline_embeddings.npy'
    DRIFTED_FILE = 'drifted_embeddings.npy'

    try:
        baseline = np.load(BASELINE_FILE)
        drifted = np.load(DRIFTED_FILE)
    except FileNotFoundError:
        print(f"Local testing requires both {BASELINE_FILE} and {DRIFTED_FILE} to be present.")
        sys.exit(1)
    
    drift_score, _ = analyze_drift(baseline, drifted)
    print(f"Local Test Complete. Score: {drift_score:.2f}%")