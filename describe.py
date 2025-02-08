import pandas as pd

class statistique():
    def __init__(self, features):
        self.features = features
        self.count = self.countf()
        self.mean = self.meanf()
        #std
        self.min = self.minf()
        #25%
        #50%
        #75%
        self.max = self.maxf()

    def countf(self):
        return len(self.features)
    def meanf(self):
        return (sum(self.features) / len(self.features))
    def minf(self):
        return(min(self.features))
    def maxf(self):
        return(max(self.features))
    

def ft_describe(features):
    stats = statistique(features)
    #print_stats


def main():
    df = pd.read_csv('datasets/dataset_train.csv')
    print(df['Birthday'])

main()