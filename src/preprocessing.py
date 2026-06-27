import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = RAW_DATA_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input directory : {RAW_DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# ========== LOAD ==========
print("Loading features: ...\n")

required_files = {
    "X": RAW_DATA_DIR / "X.npy",
    "y_plant": RAW_DATA_DIR / "y_plant.npy",
}

missing_files = [str(path) for path in required_files.values() if not path.exists()]
if missing_files:
    raise FileNotFoundError(
        "Missing extracted files from feature_extraction.py:\n"
        + "\n".join(missing_files)
    )

X = np.load(required_files["X"])
y = np.load(required_files["y_plant"])

# ========== SPLIT ==========
print("Splitting 70% train / 30% test ...\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)

print("Splitting completed \n")

# ========== SCALING ==========
print("Fitting StandardScaler ...\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = X_train_scaled.astype(dtype=np.float32)
X_test_scaled = X_test_scaled.astype(dtype=np.float32)

print("Scaling completed \n")

# ========== SAVE ==========
print(f"Saving split datasets to: {OUTPUT_DIR}\n")

# Train.
np.save(OUTPUT_DIR / "X_train.npy", X_train_scaled)
np.save(OUTPUT_DIR / "y_train.npy", y_train)

# Test
np.save(OUTPUT_DIR / "X_test.npy", X_test_scaled)
np.save(OUTPUT_DIR / "y_test.npy", y_test)

# Reproducible
joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

print("[DONE] Saved")


