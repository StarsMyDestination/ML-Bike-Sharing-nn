import NeuralNetwork as nn
import DataPreprocess as dp
import sys
import numpy as np

# Set the hyperparameters here ###
epochs = 100
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1

data_path = '../Bike-Sharing-Dataset/hour.csv'
data = dp.DataPreprocess(data_path)
N_i = data.train_features.shape[1]
network = nn.NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train': [], 'validation': []}
for e in range(epochs):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(data.train_features.index, size=128)
    for record, target in zip(data.train_features.ix[batch].values,
                              data.train_targets.ix[batch]['cnt']):
        print record.shape
        network.train(record, target)

    # Printing out the training progress
    train_loss = nn.MSE(network.run(data.train_features), data.train_targets['cnt'].values)
    val_loss = nn.MSE(network.run(data.val_features), data.val_targets['cnt'].values)
    sys.stdout.write("\rProgress: " + str(100 * e / float(epochs))
                     [:4] + "% ... Training loss: " + str(train_loss)[:5] + " ... Validation loss: " + str(val_loss)[:5])

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)
