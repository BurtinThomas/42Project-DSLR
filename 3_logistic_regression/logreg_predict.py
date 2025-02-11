import os
import sys
import pandas as pd
import numpy as np


DATASETS_LOCATION = "../datasets/"
WEIGHTS_LOCATION = "../weights/"


class Predict:
    def __init__(self, dataset):
        self.dataset = dataset
        self.weights = self.load_weights()
        self.x = None
        self.y = None
        self.load_data()
        self.preprocess_data()
        # self.predict()


    def load_weights(self):
        try:
            weights = np.load(f"{WEIGHTS_LOCATION}weights.npy", allow_pickle=True).item()
            return weights
        except Exception as e:
            print(f"Error loading weights: {e}")
            sys.exit(1)
            
        
    def load_data(self):
        try:
            data = pd.read_csv(self.dataset)
            data = data.dropna()
            self.x = np.array(data.iloc[:, 5:])
            self.y = np.array(data.loc[:, "Hogwarts House"])
        except Exception as error:
            raise RuntimeError(f"Error loading data: {error}")
        
        
    def preprocess_data(self):
        # Vérification si les données sont numériques
        if not np.issubdtype(self.x.dtype, np.number):
            print("Les données contiennent des valeurs non numériques, conversion en numériques si possible.")
            # Essayer de convertir les colonnes non numériques en numériques, si possible
            self.x = pd.to_numeric(self.x, errors='coerce')  # Les valeurs non convertibles deviendront NaN

        # Supprimer les colonnes contenant des NaN
        if np.any(np.isnan(self.x)):
            print("Des valeurs manquantes ont été trouvées dans les données, suppression des colonnes contenant des NaN.")
            self.x = self.x[:, ~np.isnan(self.x).any(axis=0)]

        # Vérification que self.x n'est pas vide après le nettoyage
        if self.x.shape[1] == 0:
            raise ValueError("Aucune donnée valide dans les features après nettoyage.")

        # Normalisation des données
        mean_x = np.mean(self.x, axis=0)
        std_x = np.std(self.x, axis=0)

        # Éviter la division par zéro : si l'écart-type est 0, mettre un petit epsilon pour éviter la division par zéro
        std_x[std_x == 0] = 1e-8

        self.x = (self.x - mean_x) / std_x

        # Traitement des labels (conversion des labels en indices)
        unique_labels = np.unique(self.y)
        self.label_mapping = {label: id for id, label in enumerate(unique_labels)}
        self.y = np.array([self.label_mapping[label] for label in self.y])


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def predict(self):
        predictions = {}
        for label, theta in self.weights.items():
            predictions[label] = self.sigmoid(np.dot(self.x, theta))

        predicted_labels = [max(predictions, key=lambda label: predictions[label][i]) for i in range(len(self.x))]
        
        print("Predictions:")
        for i, pred in enumerate(predicted_labels[:10]):
            print(f"Sample {i + 1}: Predicted label = {pred}, Actual label = {list(self.label_mapping.keys())[list(self.label_mapping.values()).index(self.y[i])]}")


def main():
    try:
        length = len(sys.argv)
        if length > 2:
            print("Usage: python script.py <dataset_filename>")
            return
        elif length == 1:
            dataset = "dataset_test.csv"
        else:
            dataset = sys.argv[1]
        Predict(f'{DATASETS_LOCATION}{dataset}')
    except Exception as error:
        print(f'{type(error).__name__}: {error}')
        
if __name__ == "__main__":
    main()
