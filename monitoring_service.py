import os
import sys
import numpy as np
import drift_detector as detector
import drift_analyzer as analyzer

# --- CONFIGURATION (DECOUPLED) ---
# We now fetch these from Environment Variables
# Syntax: os.environ.get("VARIABLE_NAME", "DEFAULT_VALUE")

NEW_DATA_PATH = os.environ.get("NEW_DATA_PATH", "/app/incoming_data")
BASELINE_PATH = os.environ.get("BASELINE_PATH", "baseline_embeddings.npy")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/status_output")

# Drift Threshold: Must convert string input to float
try:
    DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "30.0"))
except ValueError:
    print("Error: DRIFT_THRESHOLD env var is not a valid number. Using default 30.0")
    DRIFT_THRESHOLD = 30.0

# Derived paths (no need to config these separately usually)
STATUS_PATH = os.path.join(OUTPUT_DIR, "status.txt") 
SCORE_PATH = os.path.join(OUTPUT_DIR, "score.txt")

def check_for_drift():
    print(f"--- STARTING MONITORING JOB (Threshold: {DRIFT_THRESHOLD}%) ---")
    
    if not os.path.exists(BASELINE_PATH):
        print(f"CRITICAL ERROR: Baseline file {BASELINE_PATH} not found.")
        sys.exit(1)
    
    baseline = np.load(BASELINE_PATH)
    print(f"1. Loaded Baseline: {baseline.shape[0]} frames.")

    if not os.path.exists(NEW_DATA_PATH) or not os.listdir(NEW_DATA_PATH):
        print(f"Error: No data found at {NEW_DATA_PATH}. Did you mount the volume?")
        sys.exit(1)

    print("2. Loading AI Model (QC Sensor)...")
    model = detector.create_embedding_model()
    
    print(f"3. Inspecting images in {NEW_DATA_PATH}...")
    new_embeddings = detector.generate_embeddings_from_directory(model, NEW_DATA_PATH)
    
    if new_embeddings is None or new_embeddings.size == 0:
        print("Error: Could not generate embeddings from new data.")
        sys.exit(1)

    print("4. Running Statistical Analysis (QC Judge)...")
    score, _ = analyzer.analyze_drift(baseline, new_embeddings)
    
    print(f"\n>>> DRIFT SCORE: {score:.2f}%")
    print(f">>> THRESHOLD:   {DRIFT_THRESHOLD}%")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(SCORE_PATH, "w") as f:
        f.write(f"{score:.2f}")

    if score > DRIFT_THRESHOLD:
        print("\n[FAIL] HIGH DRIFT DETECTED!")
        with open(STATUS_PATH, "w") as f:
            f.write("FAIL")
        sys.exit(0) 
    else:
        print("\n[PASS] Model is operating within normal parameters.")
        with open(STATUS_PATH, "w") as f:
            f.write("PASS")
        sys.exit(0)

if __name__ == "__main__":
    check_for_drift()