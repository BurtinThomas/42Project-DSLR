import sys
import pandas as pd
import numpy as np


DATASETS_LOCATION = "../datasets/"
WEIGHTS_LOCATION = "../weights/"
LEARNING_RATE = 0.01
NUM_ITERATIONS = 1000


class Training:
    def __init__(self, dataset):
        self.dataset = dataset
        self.weights = {}
        self.x = None
        self.y = None
        self.cost_history = []

        self.load_data()
        self.preprocess_data()
        self.train()
        self.save_weights()


    def load_data(self):
        try:
            data = pd.read_csv(self.dataset)
            data = data.dropna()
            self.x = np.array((data.iloc[:,5:]))
            self.y = np.array(data.loc[:,"Hogwarts House"])
        except Exception as error:
            raise RuntimeError(f"Error loading data: {error}")


    def preprocess_data(self):
        self.x = (self.x - np.mean(self.x, axis=0)) / (np.std(self.x, axis=0) + 1e-8)

        unique_labels = np.unique(self.y)
        self.label_mapping = {label: id for id, label in enumerate(unique_labels)}
        self.y = np.array([self.label_mapping[label] for label in self.y])


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def compute_cost(self, x, y, theta):
        m = len(y)
        predictions = self.sigmoid(np.dot(x, theta))
        cost = (-1 / m) * np.sum(
            y * np.log(predictions + 1e-8) + 
            (1 - y) * np.log(1 - predictions + 1e-8)
        )
        return cost


    def gradient_descent(self, x, y):
        m, n = x.shape
        theta = np.zeros(n)
        for i in range(NUM_ITERATIONS):
            predictions = self.sigmoid(np.dot(x, theta))
            gradient = np.dot(x.T, (predictions - y)) / m # T => matrice x transposition
            theta -= LEARNING_RATE * gradient
            
            cost = self.compute_cost(self.x, self.y, theta)
            self.cost_history.append(cost)
            
        return theta

    
    def train(self):
        # bias = np.ones((self.x.shape[0], 1))
        # self.x = np.hstack((bias, self.x))
        
        unique_labels = np.unique(self.y)
        for label in unique_labels:
            y_binary = (self.y == label).astype(int)
            self.weights[label] = self.gradient_descent(self.x, y_binary)

    
    def save_weights(self):
        np.save(f"{WEIGHTS_LOCATION}weights.npy", self.weights)


def main():
    try:
        length = len(sys.argv)
        if length > 2:
            print("Usage: python script.py <dataset_filename>")
            return
        elif length == 1:
            dataset = "dataset_train.csv"
        else:
            dataset = sys.argv[1]
        Training(f'{DATASETS_LOCATION}{dataset}')
    except Exception as error:
        print(f'{type(error).__name__}: {error}')


if __name__ == "__main__":
    main()
