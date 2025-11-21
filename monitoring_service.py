import os
import sys
import numpy as np
import drift_detector as detector
import drift_analyzer as analyzer

# --- CONFIGURATION ---
NEW_DATA_PATH = "/app/incoming_data"
BASELINE_PATH = "baseline_embeddings.npy"
OUTPUT_PATH = "/app/status_output/status.txt" 
DRIFT_THRESHOLD = 30.0

def check_for_drift():
    print("--- STARTING MONITORING JOB ---")
    
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

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if score > DRIFT_THRESHOLD:
        print("\n[FAIL] HIGH DRIFT DETECTED!")
        with open(OUTPUT_PATH, "w") as f:
            f.write("FAIL")
        # FIX: Exit with 0 (Success) so the GitHub Action continues to the next step!
        sys.exit(0) 
    else:
        print("\n[PASS] Model is operating within normal parameters.")
        with open(OUTPUT_PATH, "w") as f:
            f.write("PASS")
        sys.exit(0)

if __name__ == "__main__":
    check_for_drift()