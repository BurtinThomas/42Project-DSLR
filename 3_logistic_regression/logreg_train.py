import sys
import pandas as pd
import numpy as np


DATASETS_LOCATION = "../datasets/"
LEARNING_RATE = 0.01
NUM_ITERATIONS = 1000


class Training:
    def __init__(self, dataset):
        self.dataset = dataset
        self.weights = {}
        
        self.load_data()
        self.preprocess_data()
        self.train()
        self.save_weights()
        
    
    def load_data(self):
        try:
            data = pd.read_csv(self.dataset)
            # ?
        except Exception as error:
            raise RuntimeError(f"Error loading data: {error}")
        
    def preprocess_data(self):
        pass # ?
        

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def compute_cost(self, x, y, theta):
        pass # ?

    def gradient_descent(self, x, y):
        pass # ?
    
    def train(self):
        pass #?
    
    def save_weights(self):
        pass #?


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
