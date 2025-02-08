import pandas as pd

class statistique():
    def __init__(self, features):
        self.features = features.dropna().tolist()

    def countf(self):
        return len(self.features)
    def meanf(self):
        return (sum(self.features) / len(self.features))
    #std
    def minf(self):
        return(min(self.features))
    #25%
    #50%
    #75%
    def maxf(self):
        return(max(self.features))
    

def get_informations(features):
    stats = statistique(features)
    return {
        'Count': stats.countf(),
        'Mean': stats.meanf(),
        'Min': stats.minf(),
        'Max': stats.maxf(),
    }

def ft_describe(df):
    summary = {}
    for col in df.select_dtypes(include=['number']).columns:
        summary[col] = get_informations(df[col])
    return(pd.DataFrame(summary))
    
def main():
    try:
        df = pd.read_csv('datasets/dataset_train.csv')
        print(ft_describe(df))
    except Exception as error:
        print(f'{Exception.__name__}: {error}')

main()