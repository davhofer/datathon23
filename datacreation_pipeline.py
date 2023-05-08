import pandas as pd
import numpy as np
from data import data_preprocessing, train_test_split, split_by_user
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pickle
import matplotlib.pyplot as plt

df_trainings = pd.read_csv('../trainings.csv')

# new features computed from run_logs in data.py
df_additional_features = pd.read_csv('new_features.csv')

features = data_preprocessing(df_trainings, df_additional_features, remove_type=False)

subset = [c for c in features.columns if 'recent_' not in c]

per_user_features = features[subset].copy()
per_user_features = features.drop('type', axis=1).drop('training_id', axis=1).groupby('user_id').agg(['mean', 'std', 'min', 'max', 'median', 'skew'])
per_user_features = per_user_features.reset_index()
uids = list(per_user_features['user_id'].unique())
per_user_features['user_id'] = per_user_features['user_id'].apply(lambda x: uids.index(x))
per_user_features.columns = ['_'.join(col) for col in per_user_features.columns.values]
per_user_features.rename(columns={'user_id_': 'user_id'}, inplace=True)
per_user_features_no_id = per_user_features.drop('user_id', axis=1)

# To save the uid map (int to string user_ids):
with open('uid_map_custom_metric.pickle', 'wb+') as f:
    pickle.dump(uids, f)

features['user_id'] = features['user_id'].apply(lambda x: uids.index(x))



# Define the autoencoder
class AutoEncoder(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, embedding_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
per_user_features_no_id = (per_user_features_no_id - per_user_features_no_id.mean(axis=0))/per_user_features_no_id.std(axis=0)
per_user_features_no_id = per_user_features_no_id.fillna(0)
data = torch.tensor(per_user_features_no_id.values, dtype=torch.float32)
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


n_features = per_user_features_no_id.shape[1]
embedding_size = 12
epochs = 30000
learning_rate = 1e-4


def train_autoencoder(dataloader, input_size, embedding_size, epochs, learning_rate):
    model = AutoEncoder(input_size, embedding_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # List to store the loss values for each epoch
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Calculate the average loss for this epoch
        epoch_loss /= len(dataloader)
        loss_history.append(epoch_loss)
        if epoch % 1000 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

    return model, loss_history

model, loss_history = train_autoencoder(dataloader, n_features, embedding_size, epochs, learning_rate)

embeddings = {}
for uid in per_user_features['user_id'].unique():
    x = torch.tensor(per_user_features[per_user_features['user_id'] == uid].drop('user_id', axis=1).values, dtype=torch.float32)
    emb = model.encoder(x)
    embeddings[uid] = emb.detach().numpy().reshape(-1)

df_embed = pd.DataFrame().from_dict(embeddings, orient='index')
df_embed['user_id'] = df_embed.index

def add_feature_vector(df, per_user_features):
    df = df.merge(per_user_features, on='user_id')
    return df
features_with_embeddings = add_feature_vector(features, df_embed)
# features_with_embeddings = add_feature_vector(features_with_embeddings, per_user_features)

# Save features with embeddings
features_with_embeddings.to_csv('features_with_embeddings_custom_metric.csv', index=False)