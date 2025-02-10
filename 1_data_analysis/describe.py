import sys
import pandas as pd


DATASETS_LOCATION = "../datasets/"


class statistics():
    def __init__(self, features):
        self.features = sorted(features.dropna().tolist())

    def countf(self):
        return len(self.features)
    
    def meanf(self):
        return sum(self.features) / len(self.features)
    
    def standartDeviation(self):
        variance_total = 0
        mean_value = self.meanf()
        for value in self.features:
            variance_total += (value - mean_value) ** 2
        variance = variance_total / len(self.features)
        return variance ** 0.5
    
    def minf(self):
        return min(self.features)
    
    def percentile(self, p):
        n = len(self.features)
        index = int(n * p / 100)
        if n % 2 == 0:
            return self.features[index]
        else:
            return self.features[index]

    def maxf(self):
        return max(self.features)


def get_informations(features):
    stats = statistics(features)
    return {
        'Count': stats.countf(),
        'Mean': stats.meanf(),
        'Std': stats.standartDeviation(),
        'Min': stats.minf(),
        '25%': stats.percentile(25),
        '50%': stats.percentile(50),
        '75%': stats.percentile(75),
        'Max': stats.maxf(),
    }


def ft_describe(df):
    summary = {}
    for col in df.select_dtypes(include=['number']).columns:
        summary[col] = get_informations(df[col])
    return pd.DataFrame(summary)


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
        df = pd.read_csv(f'{DATASETS_LOCATION}{dataset}')
        print(ft_describe(df))
    except Exception as error:
        print(f'{type(error).__name__}: {error}')


if __name__ == "__main__":
    main()
