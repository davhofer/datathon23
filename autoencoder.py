import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pickle
import matplotlib.pyplot as plt

# Define the autoencoder class
class AutoEncoder(nn.Module):
    """Defines a simple autoencoder using two feedforward neural networks as an encoder and decoder.
    Summarizes the input data in a lower-dimensional embedding.

    Attributes:
        input_size: An integer indicating the size of the flattened input array (number of runs * features per run).
        embedding_size: An integer indicating the size of output embedding.
    """
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

# Preprocess and prepare the data
def prepare_data(runner_data):
    """Prepares the data before its fed into the autoencoder.

    Retrieves the data from the pandas DataFrame and normalizes it.
    Then, the data is converted to a PyTorch Tensor, stacked and put in a dataloader.

    Args:
        runner_data: A list of Pandas dataframes, each one containing all runs for one runner.

    Returns:
        dataloader: A PyTorch DataLoader containing the stacked data.
    """
    all_data = []

    for df in runner_data:
        # Get the values from the DataFrame
        df = df.values

        # Normalize the data
        df = (df - df.mean()) / df.std()

        # Convert the flattened and (normalized) array to a PyTorch Tensor
        data = torch.tensor(df, dtype=torch.float32)

        all_data.append(data.unsqueeze(0))

    # Stack the tensors for all runners along the first dimension
    stacked_data = torch.cat(all_data, dim=0)

    # Create a DataLoader
    dataset = TensorDataset(stacked_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataloader

# Train the autoencoder
def train_autoencoder(dataloader, input_size, embedding_size, epochs, learning_rate):
    """Trains the autoencoder to create optimal low-dimensional embeddings.

    Args:
        dataloader: A PyTorch DataLoader containing the stacked data.
        input_size: An integer indicating the size of the flattened input array (number of runs * features per run).
        embedding_size: An integer indicating the size of output embedding.
        epochs: An integer indicating the number of epochs to train the autoencoder.
        learning_rate: A float indicating the learning rate for the optimizer.

    Returns:
        model: A PyTorch model containing the trained autoencoder.
        loss_history: A list containing the loss values for each epoch.
    """
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

# Plot the loss evolution
def plot_loss_evolution(loss_history):
    """Plots the loss evolution during training (very janky, due to fast-paced nature of hackathon).

    Args:
        loss_history: A list containing the loss values for each epoch.
    """
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss Evolution')
    plt.grid(True)
    plt.show()

def main():
    # Read the data here from runner_feat_list.pickle in the data folder
    with open('.../data/runner_feat_list.pickle', 'rb') as f:
        runner_data = pickle.load(f)

    input_size = runner_data[0].shape[1]  # Input size is the number of features
    embedding_size = 8  # Choose an appropriate size for the embeddings
    epochs = 0000
    learning_rate = 1e-4

    dataloader = prepare_data(runner_data)
    model, loss_history = train_autoencoder(dataloader, input_size, embedding_size, epochs, learning_rate)

    # Plot the loss evolution
    plot_loss_evolution(loss_history)

    embeddings = {}

    # Save the embedding for each runner
    with torch.no_grad():
        for i, df in enumerate(runner_data):
            embedding = model.encoder(torch.tensor(df.values, dtype=torch.float32))
            embeddings[i] = embedding.numpy()

    #print("Embeddings:")
    print(embeddings[0])
    print(embeddings[1])

    #save the embeddings as a pickle file
    with open('.../data/runner_embeddings.pickle', 'wb') as f:
        pickle.dump(embeddings, f)

if __name__ == "__main__":
    main()