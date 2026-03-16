# regenerate_label_encoder.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load your combined training CSV (multi_train, single_train, expert_clean)
train_df = pd.read_csv("multi_train.csv")
val_df = pd.read_csv("single_train.csv")
test_df = pd.read_csv("expert_clean.csv")

# Combine all label columns (assuming the label column is named 'label')
all_labels = pd.concat([train_df['label'], val_df['label'], test_df['label']])

# Fit label encoder
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Save inside fine_tuned_model folder
with open("fine_tuned_model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ label_encoder.pkl regenerated successfully!")
