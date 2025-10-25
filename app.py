# app.py

from flask import Flask, request, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import os

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Load Model and Tokenizer (This happens only once when the app starts) ---
MODEL_PATH = './model/'
print("Loading model and tokenizer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if model exists
if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
    print("Error: Model files not found! Please run train.py first to train and save the model.")
    # In a real app, you might want to handle this more gracefully.
    exit()

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()  # Set the model to evaluation mode

# Load the label map
with open(os.path.join(MODEL_PATH, 'label_map.json'), 'r') as f:
    id_to_label_map_str = json.load(f)
    # JSON saves keys as strings, so we convert them back to integers
    id_to_label_map = {int(k): v for k, v in id_to_label_map_str.items()}
    
print("✅ Model, tokenizer, and label map loaded successfully.")


# --- Prediction Function ---
def predict_cancer_type(sequence):
    """Predicts the cancer type for a single DNA sequence."""
    inputs = tokenizer.encode_plus(
        sequence,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    prediction_idx = torch.argmax(logits, dim=1).item()
    predicted_label = id_to_label_map[prediction_idx]
    
    probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
    confidence = probabilities[prediction_idx]
    
    return predicted_label, confidence

# --- Define Website Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    if request.method == 'POST':
        # Get the sequence from the form
        sequence = request.form.get('sequence')
        if sequence:
            label, confidence = predict_cancer_type(sequence)
            prediction_result = {
                'label': label,
                'confidence': f"{confidence:.2%}",
                'sequence': sequence
            }
    # Render the HTML page
    return render_template('index.html', result=prediction_result)

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)