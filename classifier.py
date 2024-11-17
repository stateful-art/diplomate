import sys
import json
import torch
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the saved model and tokenizer
# MODEL_PATH = "/app/model/game_text_classifier_model"
# TOKENIZER_PATH = "/app/model/game_text_classifier_model"

MODEL_PATH = "output/diplomatic_text_classifier_model"
TOKENIZER_PATH = "output/diplomatic_text_classifier_model"


# Check if the model and tokenizer files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model directory not found at {MODEL_PATH}")
if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
    raise FileNotFoundError(f"Model config file not found in {MODEL_PATH}")
if not os.path.exists(os.path.join(TOKENIZER_PATH, "tokenizer_config.json")):
    raise FileNotFoundError(f"Tokenizer config file not found in {TOKENIZER_PATH}")

# Load model and tokenizer
try:
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True, ignore_mismatched_sizes=True)
    tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    sys.exit(1)

# Define label mapping

id_to_label = {
    0: "cooperation",
    1: "negotiation",
    2: "alliance_proposal",
    3: "threat",
    4: "intimidation",
    5: "compromise",
    6: "peace_offer",
    7: "declaration_of_war",
    8: "ceasefire_request",
    9: "trade_proposal",
    10: "intelligence_sharing",
    11: "diplomatic_pressure",
    12: "sanctions_threat",
    13: "mediation_offer",
    14: "neutrality_declaration",
    15: "territorial_claim",
    16: "diplomatic_protest",
    17: "apology",
    18: "praise_or_commendation",
    19: "criticism",
    20: "request_for_aid",
    21: "offer_of_assistance",
    22: "ultimatum",
    23: "non_aggression_pact",
    24: "treaty_proposal",
    25: "diplomatic_recognition",
    26: "severance_of_relations",
    27: "espionage_accusation",
    28: "denial_of_accusations",
    29: "call_for_unity",
    30: "appeal_to_international_law",
    31: "economic_cooperation",
    32: "cultural_exchange",
    33: "military_cooperation",
    34: "humanitarian_aid_offer",
    35: "request_for_mediation",
    36: "diplomatic_immunity_invocation",
    37: "extradition_request",
    38: "asylum_offer",
    39: "propaganda",
    40: "disinformation",
    41: "confidence_building_measure",
    42: "arms_control_proposal",
    43: "environmental_cooperation",
    44: "technology_transfer",
    45: "diplomatic_demarche",
    46: "formal_complaint",
    47: "request_for_clarification",
    48: "expression_of_concern",
    49: "congratulatory_message",
    50: "condolences",
    51: "neutral_statement",
    52: "procedural_communication",
    53: "information_request",
    54: "summit_proposal",
    55: "arbitration_request",
    56: "border_dispute_resolution",
    57: "diplomatic_crisis_management",
    58: "economic_sanctions_announcement",
    59: "humanitarian_corridor_request",
    60: "peacekeeping_mission_proposal",
    61: "condemnation",
}
def predict_single(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_id = outputs.logits.argmax().item()
    return id_to_label[predicted_class_id]

def predict_batch(texts):
    results = []
    for text in texts:
        label = predict_single(text)
        results.append({"text": text, "label": label})
    return results

if __name__ == "__main__":
    try:
        # Read JSON input from stdin
        input_json = sys.stdin.read()
        
        try:
            input_texts = json.loads(input_json)
        except json.JSONDecodeError:
            print("Error: Invalid JSON input")
            sys.exit(1)

        if not isinstance(input_texts, list):
            print("Error: Input must be a JSON array of strings")
            sys.exit(1)

        predictions = predict_batch(input_texts)
        
        # Handle broken pipe error when printing predictions
        try:
            for prediction in predictions:
                print(json.dumps(prediction, ensure_ascii=False))
        except BrokenPipeError:
            # Python flushes standard streams on exit; redirect remaining output
            # to devnull to avoid another BrokenPipeError at shutdown
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            sys.exit(1)  # Python exits with error code 1 on EPIPE
            
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Explicitly flush and close stdout to avoid BrokenPipeError during cleanup
        try:
            sys.stdout.flush()
        except BrokenPipeError:
            pass
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass