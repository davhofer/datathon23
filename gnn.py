import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
from sklearn.preprocessing import StandardScaler
import random

# Load the data
dataframe = pd.read_csv('/Users/luc/Documents/Coding Adventures/datathon/data/features_with_embeddings.csv')

# Assuming you have a pandas DataFrame called 'dataframe'
# Drop the 'user_id' column and convert the DataFrame to a tensor
# Encode the 'type' column to integers
def normalize_features(df, columns_to_exclude=['user_id', 'type']):
    scaler = StandardScaler()
    columns_to_normalize = [col for col in df.columns if col not in columns_to_exclude]
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df, scaler

def create_mask(num_nodes, num_masked_nodes):
    mask_indices = random.sample(range(num_nodes), num_masked_nodes)
    mask = torch.zeros(num_nodes, dtype=bool)
    mask[mask_indices] = True
    return mask

def encode_labels(df, label_col='type'):
    le = LabelEncoder()
    encoded_labels = le.fit_transform(df[label_col])
    return encoded_labels

def dataframe_to_tensor(df, label_col='type'):
    labels = torch.tensor(encode_labels(df, label_col), dtype=torch.long)
    features = torch.tensor(df.drop(columns=['user_id', label_col]).values, dtype=torch.float)
    return features, labels

def create_fully_connected_edges(num_nodes):
    return torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t().contiguous()

def remove_nan_type_nodes(user_data):
    return user_data.dropna(subset=['type'])

def sort_by_time(user_data):
    return user_data.sort_values(by=['month', 'weekday', 'hour'])

def split_data_by_time(user_data, train_ratio=1.0):
    n = len(user_data)
    train_size = int(n * train_ratio)
    train_data = user_data.iloc[:train_size]
    test_data = user_data.iloc[train_size:]
    return train_data, test_data

# Group the DataFrame by 'user_id'
grouped_data = dataframe.groupby('user_id')

train_data_list = []
test_data_list = []
num_masked_nodes = 3
for user_id, user_data in grouped_data:
    user_data = remove_nan_type_nodes(user_data)
    user_data = sort_by_time(user_data)
    user_data, _ = normalize_features(user_data)

    if not user_data.empty:
        x, y = dataframe_to_tensor(user_data)
        edge_index = create_fully_connected_edges(len(user_data))
        mask = create_mask(len(user_data), num_masked_nodes)

        data = Data(x=x, edge_index=edge_index, y=y, mask=mask)
        train_data_list.append(data)
        test_data_list.append(data.clone())

# Create DataLoaders for train and test sets
batch_size = 8
train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)

# Define the model
class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads, concat=True, dropout=0.5)
        self.conv2 = GATConv(hidden_channels * num_heads, num_classes, heads=1, concat=True, dropout=0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
# Train
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set parameters
num_features = train_data_list[0].num_features
hidden_channels = 64
num_classes = len(torch.unique(torch.cat([data.y for data in train_data_list])))
num_epochs = 50
learning_rate = 0.0005

# Initialize the GAT model, optimizer, and move them to the device
model = GAT(num_features, hidden_channels, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    correct_train = 0
    total_train = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        # Apply the mask during training
        masked_out = out[~data.mask]
        masked_y = data.y[~data.mask]

        loss = F.nll_loss(masked_out, masked_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate training accuracy
        _, pred_train = masked_out.max(dim=1)
        correct_train += int((pred_train == masked_y).sum())
        total_train += masked_y.size(0)
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}, Train Accuracy: {100 * correct_train / total_train:.2f}%")

# Test the model
model.eval()
correct = 0
total = 0
for data in test_loader:
    data = data.to(device)
    out = model(data.x, data.edge_index)

    # Calculate test accuracy only on masked nodes
    masked_out_test = out[data.mask]
    masked_y_test = data.y[data.mask]
    _, pred_test = masked_out_test.max(dim=1)
    correct += int((pred_test == masked_y_test).sum())
    total += masked_y_test.size(0)

print("Test Accuracy: {:.2f}%".format(100 * correct / total))

# Save the model
torch.save(model.state_dict(), 'model_1.pt')