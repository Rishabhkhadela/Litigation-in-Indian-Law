import pandas as pd

# Load your expert.csv
df = pd.read_csv("expert.csv")

print("🔍 Original columns:", df.columns)
print("🔍 Example row:", df.iloc[0])

# ---- FIX STEP ----
# If your data has a 'text' column already, keep it.
# If not, we need to extract text from one of the nested dict/array columns.

# Case 1: If the CSV has a column with nested dicts/arrays, flatten it:
if "text" not in df.columns:
    # Try to use one of the rank columns (rank2, rank3, etc.)
    # Adjust depending on which one contains the actual case text.
    df["text"] = df["rank2"].astype(str)  

# Keep only text + label
clean_df = df[["text", "label"]]

# Save cleaned file
clean_df.to_csv("expert_clean.csv", index=False)

print("✅ Cleaned expert.csv saved as expert_clean.csv with shape:", clean_df.shape)
