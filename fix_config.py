from transformers import BertConfig
import os

MODEL_DIR = "fine_tuned_model"
config_path = os.path.join(MODEL_DIR, "config.json")

print(f"🔧 Fixing config.json in {MODEL_DIR} ...")

# Adjust num_labels to match your classification task
config = BertConfig.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=2)

# Save valid config.json
config.save_pretrained(MODEL_DIR)

print(f"✅ Fixed! New config.json written at {config_path}")
