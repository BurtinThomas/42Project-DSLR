import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, weights, biases):
    predictions = {}
    for house in weights:
        z = np.dot(X, weights[house]) + biases[house]
        predictions[house] = sigmoid(z)
    return max(predictions, key=predictions.get)

def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model['weights'], model['biases']

def main():
    df = pd.read_csv('../datasets/dataset_test.csv')
    df = df.drop('Hogwarts House', axis=1)
    df = df.dropna()

    numeric_columns = [col for col in df.select_dtypes(include=['number']).columns if col != 'Index']
    x = df[numeric_columns].values

    scaler = StandardScaler()
    X = scaler.fit_transform(x)

    weights, biases = load_model('parametre.pkl')
    y_pred = [predict(x, weights, biases) for x in X]
    result_df = pd.DataFrame(y_pred)
    print(result_df)

main()