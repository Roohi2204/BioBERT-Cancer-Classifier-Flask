# train.py

import torch
import pandas as pd
import numpy as np
import random
import json
import os
from sklearn.model_selection import train_test_split
# train.py (FIXED)
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW # Import AdamW from PyTorch's optim module
from torch.utils.data import DataLoader, Dataset

# --- All the code from the Colab notebook for data generation and the Dataset class goes here ---

def create_synthetic_cancer_data(num_samples=1000):
    """Generates a pandas DataFrame with synthetic DNA sequences and cancer types."""
    print("🧬 Generating synthetic DNA sequence data...")
    # (Same function as before)
    data = []
    cancer_types = ['BRCA', 'LUNG', 'COAD'] # Breast, Lung, Colon Adenocarcinoma
    bases = ['A', 'C', 'G', 'T']
    for _ in range(num_samples):
        cancer_type = random.choice(cancer_types)
        seq_length = random.randint(100, 200)
        sequence = ''.join(random.choice(bases) for _ in range(seq_length))
        if cancer_type == 'BRCA':
            sequence = sequence.replace('A'*2, 'ACGT', 1)
        elif cancer_type == 'LUNG':
            sequence = sequence.replace('C'*2, 'GATTACA', 1)
        else: # COAD
            sequence = sequence.replace('G'*2, 'TTAG', 1)
        data.append({'sequence': sequence, 'cancer_type': cancer_type})
    df = pd.DataFrame(data)
    print(f"✅ Generated {len(df)} samples.")
    return df

class CancerSequenceDataset(Dataset):
    # (Same class as before)
    def __init__(self, sequences, labels, tokenizer, max_len):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, item):
        sequence = str(self.sequences[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(sequence, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(label, dtype=torch.long)}


# --- Main Training Logic ---
def train_and_save_model():
    # Configuration
    MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    SAVE_PATH = './model/'

    # 1. Prepare Data
    data_frame = create_synthetic_cancer_data()
    label_map = {name: i for i, name in enumerate(data_frame['cancer_type'].unique())}
    id_to_label_map = {i: name for name, i in label_map.items()}
    data_frame['label'] = data_frame['cancer_type'].map(label_map)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    X_train, _, y_train, _ = train_test_split(
        data_frame['sequence'].tolist(), 
        data_frame['label'].tolist(), 
        test_size=0.1, random_state=42, stratify=data_frame['label'].tolist()
    )
    
    train_dataset = CancerSequenceDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_map))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    print("\n🚀 Starting model training...")
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"--- Epoch {epoch + 1}/{EPOCHS} complete ---")

    # 4. Save the fine-tuned model, tokenizer, and label map
    print("\n💾 Saving model and tokenizer...")
    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    with open(os.path.join(SAVE_PATH, 'label_map.json'), 'w') as f:
        json.dump(id_to_label_map, f)

    print("✅ Training complete and all files saved to the 'model' directory.")

if __name__ == '__main__':
    train_and_save_model()