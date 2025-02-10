import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradients(X, y, weights, bias):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights) + bias)
    dw = (1/m) * np.dot(X.T, (predictions - y))
    db = (1/m) * np.sum(predictions - y)
    return dw, db

def gradient_descent(X, y, weights, bias, learning_rate, iterations):
    for i in range(iterations):
        dw, db = compute_gradients(X, y, weights, bias)
        weights -= learning_rate * dw
        bias -= learning_rate * db
    return weights, bias

def one_vs_rest(X, y, learning_rate, iterations):
    weights = {}
    biases = {}
    for house in list(set(y)):
        y_binary = np.array([1 if label == house else 0 for label in y])
        w = np.zeros(X.shape[1])
        b = 0
        w, b = gradient_descent(X, y_binary, w, b, learning_rate, iterations)
        weights[house] = w
        biases[house] = b
    return weights, biases

def predict(X, weights, biases):
    predictions = {}
    for house in weights:
        z = np.dot(X, weights[house]) + biases[house]
        predictions[house] = sigmoid(z)
    return max(predictions, key=predictions.get)

def evaluate_performance(X, y, weights, biases):
    y_pred = [predict(x, weights, biases) for x in X]
    accuracy = accuracy_score(y, y_pred)
    return accuracy

def save_model(weights, biases, model_file):
    model = {'weights': weights, 'biases': biases}
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

def main():
    try:
        learning_rate = 0.025
        iterations = 500
        df = pd.read_csv('../datasets/dataset_train.csv')
        df = df.dropna()
        
        numeric_columns = [col for col in df.select_dtypes(include=['number']).columns if col != 'Index']
        x = df[numeric_columns].values
        y = df['Hogwarts House'].values

        scaler = StandardScaler()
        X = scaler.fit_transform(x)
        
        weights, biases = one_vs_rest(X, y, learning_rate, iterations)
        accuracy = evaluate_performance(X, y, weights, biases)
        print(accuracy)
        save_model(weights, biases, 'parametre.pkl', )
    except Exception as error:
        print(f"{type(error).__name__} : {error}")
    except KeyboardInterrupt:
        print("\nInput interrupted. Exiting...")

if __name__ == "__main__":
    main()