from sklearn.datasets import load_breast_cancer
import pandas as pd
from pathlib import Path


# Load the dataset
data = load_breast_cancer()

# Create DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df["target"] = data.target

# Create data directory if it doesn't exist
Path("data").mkdir(exist_ok=True)

# Save dataset
df.to_csv("data/breast_cancer.csv", index=False)

print("Dataset created successfully!")
print(f"Shape: {df.shape}")
print(f"Saved to: data/breast_cancer.csv")
print("\nFirst 5 rows:")
print(df.head())