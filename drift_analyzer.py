import numpy as np
from scipy.stats import ks_2samp
import os

# --- CONFIGURATION ---
BASELINE_FILE = 'baseline_embeddings.npy'
DRIFTED_FILE = 'drifted_embeddings.npy'
P_VALUE_THRESHOLD = 0.05 
# The p-value threshold determines statistical significance. 
# A result where p < 0.05 means there is a strong statistical difference (DRIFT).

def load_embeddings():
    """Loads the baseline and drifted embeddings from the .npy files."""
    try:
        # In the production monitor, these files are loaded from the root /app folder.
        baseline = np.load(BASELINE_FILE)
        drifted = np.load(DRIFTED_FILE)
        
        print(f"Loaded Baseline Embeddings: {baseline.shape}")
        print(f"Loaded Drifted Embeddings: {drifted.shape}")
        
        # Ensure the embeddings have the same number of features (1280 for MobileNetV2)
        if baseline.shape[1] != drifted.shape[1]:
            raise ValueError("Embedding feature counts do not match!")
            
        return baseline, drifted
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found. Have you run drift_detector.py? {e}")
        return None, None
    except ValueError as e:
        print(f"Error: {e}")
        return None, None

def analyze_drift(baseline, drifted):
    """
    Performs the Kolmogorov-Smirnov (KS) two-sample test feature-by-feature.
    The drift score is the percentage of features that show significant statistical difference.
    """
    num_features = baseline.shape[1]
    significant_drift_count = 0
    p_values = [] # Not used in production, but good to collect

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
    # This block is for local development/testing only
    baseline, drifted = load_embeddings()
    
    if baseline is not None and drifted is not None:
        drift_score, _ = analyze_drift(baseline, drifted)
        print(f"Local Test Complete. Score: {drift_score:.2f}%")