import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from tqdm import tqdm
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the existing data
def load_data(file_path):
    return pd.read_csv(file_path)

# Prepare the data
data = load_data('input/diplomacy_data_full.csv')
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Convert to Dataset objects
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=62).to(device)

# Create label to id mapping
label_to_id = {
    "cooperation": 0,
    "negotiation": 1,
    "alliance_proposal": 2,
    "threat": 3,
    "intimidation": 4,
    "compromise": 5,
    "peace_offer": 6,
    "declaration_of_war": 7,
    "ceasefire_request": 8,
    "trade_proposal": 9,
    "intelligence_sharing": 10,
    "diplomatic_pressure": 11,
    "sanctions_threat": 12,
    "mediation_offer": 13,
    "neutrality_declaration": 14,
    "territorial_claim": 15,
    "diplomatic_protest": 16,
    "apology": 17,
    "praise_or_commendation": 18,
    "criticism": 19,
    "request_for_aid": 20,
    "offer_of_assistance": 21,
    "ultimatum": 22,
    "non_aggression_pact": 23,
    "treaty_proposal": 24,
    "diplomatic_recognition": 25,
    "severance_of_relations": 26,
    "espionage_accusation": 27,
    "denial_of_accusations": 28,
    "call_for_unity": 29,
    "appeal_to_international_law": 30,
    "economic_cooperation": 31,
    "cultural_exchange": 32,
    "military_cooperation": 33,
    "humanitarian_aid_offer": 34,
    "request_for_mediation": 35,
    "diplomatic_immunity_invocation": 36,
    "extradition_request": 37,
    "asylum_offer": 38,
    "propaganda": 39,
    "disinformation": 40,
    "confidence_building_measure": 41,
    "arms_control_proposal": 42,
    "environmental_cooperation": 43,
    "technology_transfer": 44,
    "diplomatic_demarche": 45,
    "formal_complaint": 46,
    "request_for_clarification": 47,
    "expression_of_concern": 48,
    "congratulatory_message": 49,
    "condolences": 50,
    "neutral_statement": 51,
    "procedural_communication": 52,
    "information_request": 53,
    "summit_proposal": 54,
    "arbitration_request": 55,
    "border_dispute_resolution": 56,
    "diplomatic_crisis_management": 57,
    "economic_sanctions_announcement": 58,
    "humanitarian_corridor_request": 59,
    "peacekeeping_mission_proposal": 60,
    "condemnation": 61,
}

id_to_label = {v: k for k, v in label_to_id.items()}

# Tokenize function
def tokenize_and_encode_labels(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokenized["label"] = [label_to_id[label] for label in examples["label"]]
    return tokenized

# Apply tokenization
tokenized_train = train_dataset.map(tokenize_and_encode_labels, batched=True, remove_columns=train_dataset.column_names)
tokenized_test = test_dataset.map(tokenize_and_encode_labels, batched=True, remove_columns=test_dataset.column_names)

# Custom metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Set up training arguments
output_dir = 'output/results'
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='output/logs',
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="tensorboard",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# Train the model
print("Starting model training...")
trainer.train()
print("Model training completed.")

# Evaluate the model
print("Evaluating the model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Find the latest checkpoint
checkpoints = [dir for dir in os.listdir(output_dir) if dir.startswith('checkpoint-')]
latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
latest_checkpoint_path = os.path.join(output_dir, latest_checkpoint)

print(f"Latest checkpoint: {latest_checkpoint_path}")

# Load the best model
best_model = DistilBertForSequenceClassification.from_pretrained(latest_checkpoint_path).to(device)

# Function to classify new text
def classify_text(text, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)
    return id_to_label[prediction.item()]

# Example usage
try:
    new_text = "We propose a comprehensive trade agreement to strengthen our economic ties."
    result = classify_text(new_text, best_model)
    print(f"The text '{new_text}' is classified as: {result}")
except Exception as e:
    print(f"An error occurred during classification: {e}")

# Save the best model and tokenizer to the model directory
final_output_dir = "output/diplomatic_text_classifier_model"
best_model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)

print(f"\nBest model and tokenizer saved to: {final_output_dir}")
print("\nYou can load the model and tokenizer later with:")
print(f"model = DistilBertForSequenceClassification.from_pretrained('{final_output_dir}')")
print(f"tokenizer = DistilBertTokenizer.from_pretrained('{final_output_dir}')")

# Verify saved files
print("\nSaved files:")
if os.path.exists(final_output_dir):
    print(f"\nContents of {final_output_dir}:")
    for file in os.listdir(final_output_dir):
        print(f" - {file}")
else:
    print(f"\n{final_output_dir} does not exist.")

print("\nTo view training progress and metrics in TensorBoard, run:")
print("tensorboard --logdir output/logs")
print("Then open the provided URL in your web browser.")