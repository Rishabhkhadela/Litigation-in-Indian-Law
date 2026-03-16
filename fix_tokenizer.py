from transformers import AutoTokenizer

# Use the same base model you trained on
BASE_MODEL = "nlpaueb/legal-bert-base-uncased"  # if you trained with InLegalBERT
# or BASE_MODEL = "bert-base-uncased"

# Load base tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Save tokenizer into your fine-tuned model folder
tokenizer.save_pretrained("F:/litigation_predictor/fine_tuned_model")

print("✅ Tokenizer files regenerated in fine_tuned_model folder.")
