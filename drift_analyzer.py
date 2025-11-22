import numpy as np
from scipy.stats import ks_2samp
import os

# --- CONFIGURATION (DECOUPLED) ---
# P-Value Threshold: The strictness of the statistical test
try:
    P_VALUE_THRESHOLD = float(os.environ.get("P_VALUE_THRESHOLD", "0.05"))
except ValueError:
    print("Error: P_VALUE_THRESHOLD env var is not valid. Using default 0.05")
    P_VALUE_THRESHOLD = 0.05

def analyze_drift(baseline, drifted):
    """
    Performs the Kolmogorov-Smirnov (KS) two-sample test feature-by-feature.
    """
    if baseline.shape[1] != drifted.shape[1]:
        raise ValueError("Embedding feature counts do not match!")
        
    num_features = baseline.shape[1]
    significant_drift_count = 0
    p_values = [] 

    # 1. Iterate through every feature
    for i in range(num_features):
        baseline_feature = baseline[:, i]
        drifted_feature = drifted[:, i]
        
        # 2. Perform KS test
        _, p_value = ks_2samp(baseline_feature, drifted_feature)
        p_values.append(p_value)

        # 3. Check for drift using the Configured Threshold
        if p_value < P_VALUE_THRESHOLD:
            significant_drift_count += 1

    # Calculate the drift score
    drift_score = (significant_drift_count / num_features) * 100

    print("\n--- Summary of Drift Detection ---")
    print(f"P-Value Threshold used: {P_VALUE_THRESHOLD}")
    print(f"Total features analyzed: {num_features}")
    print(f"Features showing significant drift: {significant_drift_count}")
    print(f"Overall Drift Score: {drift_score:.2f}%")
    
    return drift_score, p_values

if __name__ == "__main__":
    # Local testing block (unchanged logic, just ensuring imports work)
    # ... (You can leave the local test block as is or update it to match)
    pass