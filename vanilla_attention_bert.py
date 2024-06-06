import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertModel, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Check if a GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the IMDB review dataset from CSV
df = pd.read_csv('train.csv')

# Prepare the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and encode the input sequences
tokens = tokenizer.batch_encode_plus(
    df['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors='pt'
)

input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']
labels = torch.tensor(df['sentiment'].map({'pos': 1, 'neg': 0}).tolist())

# Split the data into train and validation sets
train_input_ids, valid_input_ids, train_attention_mask, valid_attention_mask, train_labels, valid_labels = train_test_split(
    input_ids.cpu(),
    attention_mask.cpu(),
    labels.cpu(),
    test_size=0.2,
    random_state=42,
    stratify=labels.cpu()
)

train_input_ids = train_input_ids.to(device)
valid_input_ids = valid_input_ids.to(device)
train_attention_mask = train_attention_mask.to(device)
valid_attention_mask = valid_attention_mask.to(device)
train_labels = train_labels.to(device)
valid_labels = valid_labels.to(device)

# Define batch size
batch_size = 32

# Create TensorDatasets for train and validation sets
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
valid_dataset = TensorDataset(valid_input_ids, valid_attention_mask, valid_labels)

# Create DataLoaders for train and validation sets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

# Define the Attention class
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        # input dimensions: (batch_size, seq_length, hidden_size)
        q = self.query(input)
        k = self.key(input)
        v = self.value(input)
        # q, k, v dimensions: (batch_size, seq_length, hidden_size)

        attention_weights = torch.bmm(q, k.transpose(1, 2))
        # attention_weights dimensions: (batch_size, seq_length, seq_length)
        attention_weights = self.softmax(attention_weights)

        output = torch.bmm(attention_weights, v)
        # output dimensions: (batch_size, seq_length, hidden_size)
        return output

# Instantiate the BERT-based binary classification model with Attention
class BERTBinaryClassifier(nn.Module):
    def __init__(self, hidden_size):
        super(BERTBinaryClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        # pooled_output dimensions: (batch_size, hidden_size)

        attention_output = self.attention(pooled_output.unsqueeze(1))
        # attention_output dimensions: (batch_size, 1, hidden_size)

        logits = self.fc(attention_output.squeeze(1))
        # logits dimensions: (batch_size, num_classes)

        return logits

model = BERTBinaryClassifier(hidden_size=768).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Train the model
model.train()
for epoch in range(5):  # Perform 5 epochs
    for batch in train_dataloader:
        batch_input_ids, batch_attention_mask, batch_labels = batch
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        logits = model(batch_input_ids, batch_attention_mask)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        valid_loss = 0.0
        valid_predictions = []
        valid_true_labels = []

        for batch in valid_dataloader:
            batch_input_ids, batch_attention_mask, batch_labels = batch
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_input_ids, batch_attention_mask)
            batch_loss = criterion(logits, batch_labels)
            valid_loss += batch_loss.item()

            batch_predictions = torch.argmax(logits, dim=1)
            valid_predictions.extend(batch_predictions.cpu().tolist())
            valid_true_labels.extend(batch_labels.cpu().tolist())

        valid_loss /= len(valid_dataloader)
        valid_accuracy = accuracy_score(valid_true_labels, valid_predictions)
        valid_precision = precision_score(valid_true_labels, valid_predictions)
        valid_recall = recall_score(valid_true_labels, valid_predictions)
        valid_f1 = f1_score(valid_true_labels, valid_predictions)

    model.train()

    print(f"Epoch {epoch + 1}:")
    print(f"Train Loss: {loss.item():.4f} | Validation Loss: {valid_loss:.4f}")
    print(f"Validation Accuracy: {valid_accuracy:.4f}")
    print(f"Validation Precision: {valid_precision:.4f}")
    print(f"Validation Recall: {valid_recall:.4f}")
    print(f"Validation F1-Score: {valid_f1:.4f}")
    print("--------------------")